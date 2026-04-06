from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set, Tuple

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nlp = spacy.load("en_core_web_sm")


@dataclass
class RuleResult:
    name: str
    triggered: bool
    weight: int
    priority: str
    reason: str
    tags: List[str] = field(default_factory=list)
    hard_override: bool = False


@dataclass
class AnalysisResult:
    label: str
    score: int
    threshold: int
    intent: str
    hard_override_applied: bool
    triggered_rules: List[RuleResult]


BASE_AMBIGUITY_THRESHOLD = 4

ACTION_VERBS = {
    "book", "order", "send", "delete", "schedule", "fix", "buy", "cancel", "reserve", "arrange"
}
SCHEDULING_VERBS = {"book", "schedule", "reserve", "arrange", "plan"}
LOCATION_REQUIRED_VERBS = {"book", "reserve", "schedule", "meet", "arrange"}
QUANTITY_REQUIRED_VERBS = {"buy", "order"}
_BASE_VAGUE = {"it", "this", "that", "something", "someone", "thing"}

def _build_vague_set() -> Set[str]:
    expanded = set(_BASE_VAGUE)
    seed_words = ["thing", "stuff", "somewhere", "someone", "something"]
    for word in seed_words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                w = lemma.name().lower().replace("_", " ")
                if len(w) > 2:
                    expanded.add(w)
    return expanded

VAGUE_PRONOUNS: Set[str] = _build_vague_set()

QUESTION_STARTERS = {
    "what", "how", "why", "who", "where", "when", "which", "whom",
    "did", "does", "do", "is", "are", "can", "could", "would", "will", "should"
}

TEMPORAL_WORDS = {
    "today", "tomorrow", "tonight", "morning", "evening", "afternoon",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "next", "this", "week", "month", "year", "am", "pm"
}

INTENT_REQUIRED_PARAMS: Dict[str, Set[str]] = {
    "booking": {"object", "datetime", "location"},
    "purchase": {"object", "quantity"},
    "task": {"object"},
}

INTENT_THRESHOLD_OFFSET: Dict[str, int] = {
    "booking": -2,
    "purchase": 0,
    "task": 0,
    "informational": 2,
    "unknown": 0,
}


@dataclass
class IntentProfile:
    name: str
    required_params: Set[str]
    detected_params: Set[str]
    missing_params: Set[str]


def _contains_task_action(doc, token_list: List[str]) -> bool:
    return any(_is_task_intent_token(token, ACTION_VERBS) for token in doc) or _starts_with_intent_verb(
        token_list,
        ACTION_VERBS,
    )


def _starts_with_intent_verb(token_list: List[str], verb_set) -> bool:
    if not token_list:
        return False
    return token_list[0] in verb_set


def _is_task_intent_token(token, verb_set) -> bool:
    if token.lemma_.lower() not in verb_set:
        return False
    if token.pos_ == "VERB":
        return True
    return token.dep_ == "ROOT"


def _has_location_signal(doc, token_list: List[str]) -> bool:
    has_location_entity = any(ent.label_ in {"GPE", "LOC", "FAC"} for ent in doc.ents)
    has_location_phrase = any(
        token.dep_ == "prep" and token.lower_ in {"at", "in", "to", "near"}
        for token in doc
    )
    # Fallback: bare PROPN after a comma or sentence boundary (e.g. "..., Hyderabad")
    has_propn_fallback = any(
        token.pos_ == "PROPN" and token.i > 0 and doc[token.i - 1].text in {",", ".", "?"}
        for token in doc
    )
    return has_location_entity or has_location_phrase or has_propn_fallback or "where" in token_list


def _has_temporal_signal(doc, token_list: List[str]) -> bool:
    has_temporal_entity = any(ent.label_ in {"DATE", "TIME"} for ent in doc.ents)
    has_temporal_token = any(word in TEMPORAL_WORDS for word in token_list)
    return has_temporal_entity or has_temporal_token or "when" in token_list


def _has_quantity_signal(doc, token_list: List[str]) -> bool:
    has_numeric_token = any(token.like_num for token in doc)
    quantity_keywords = {"some", "few", "many", "several", "couple", "pair", "dozen"}
    has_quantity_keyword = any(word in quantity_keywords for word in token_list)
    return has_numeric_token or has_quantity_keyword


def _has_object_signal(doc) -> bool:
    return any(token.pos_ in {"NOUN", "PROPN"} for token in doc)


def detect_intent(doc, token_list: List[str]) -> str:
    if token_list and token_list[0] in QUESTION_STARTERS and not _contains_task_action(doc, token_list):
        return "informational"

    has_scheduling = any(_is_task_intent_token(token, SCHEDULING_VERBS) for token in doc) or _starts_with_intent_verb(
        token_list,
        SCHEDULING_VERBS,
    )
    if has_scheduling:
        return "booking"

    has_purchase = any(_is_task_intent_token(token, QUANTITY_REQUIRED_VERBS) for token in doc) or _starts_with_intent_verb(
        token_list,
        QUANTITY_REQUIRED_VERBS,
    )
    if has_purchase:
        return "purchase"

    if _contains_task_action(doc, token_list):
        return "task"

    return "unknown"


def build_intent_profile(doc, token_list: List[str]) -> IntentProfile:
    intent = detect_intent(doc, token_list)

    detected_params: Set[str] = set()
    if _has_object_signal(doc):
        detected_params.add("object")
    if _has_temporal_signal(doc, token_list):
        detected_params.add("datetime")
    if _has_location_signal(doc, token_list):
        detected_params.add("location")
    if _has_quantity_signal(doc, token_list):
        detected_params.add("quantity")

    required_params = INTENT_REQUIRED_PARAMS.get(intent, set())
    missing_params = required_params - detected_params
    return IntentProfile(
        name=intent,
        required_params=required_params,
        detected_params=detected_params,
        missing_params=missing_params,
    )


def _dynamic_threshold(intent: str) -> int:
    return max(2, BASE_AMBIGUITY_THRESHOLD + INTENT_THRESHOLD_OFFSET.get(intent, 0))


def _parameter_completeness_rules(profile: IntentProfile) -> List[RuleResult]:
    results: List[RuleResult] = []

    if profile.name == "booking" and "datetime" in profile.missing_params:
        results.append(
            RuleResult(
                name="MissingRequiredDateTime",
                triggered=True,
                weight=3,
                priority="high",
                reason="Booking/scheduling intent is missing required date/time parameter.",
                tags=["parameter", "intent-critical"],
                hard_override=True,
            )
        )

    if profile.name == "booking" and "location" in profile.missing_params:
        results.append(
            RuleResult(
                name="MissingRequiredLocation",
                triggered=True,
                weight=2,
                priority="medium",
                reason="Booking/scheduling intent is missing required location parameter.",
                tags=["parameter", "intent-required"],
            )
        )

    if profile.name in {"booking", "purchase", "task"} and "object" in profile.missing_params:
        results.append(
            RuleResult(
                name="MissingRequiredObject",
                triggered=True,
                weight=3,
                priority="high",
                reason=f"{profile.name.capitalize()} intent is missing required object parameter.",
                tags=["parameter", "intent-critical"],
                hard_override=(profile.name in {"booking", "task"}),
            )
        )

    if profile.name == "purchase" and "quantity" in profile.missing_params:
        results.append(
            RuleResult(
                name="MissingRequiredQuantity",
                triggered=True,
                weight=2,
                priority="medium",
                reason="Purchase intent is missing quantity parameter.",
                tags=["parameter", "intent-required"],
            )
        )

    return results


def rule_vague_pronoun(doc, token_list: List[str]) -> RuleResult:
    for token in doc:
        if token.lower_ in VAGUE_PRONOUNS:
            if token.lower_ in {"this", "that"} and token.dep_ == "det" and token.head.pos_ == "NOUN":
                continue
            return RuleResult(
                name="VaguePronoun",
                triggered=True,
                weight=2,
                priority="medium",
                reason=f"Contains vague referent '{token.text}' without explicit context.",
                tags=["reference"],
            )
    return RuleResult("VaguePronoun", False, 2, "medium", "", tags=["reference"])


def rule_action_without_object(doc, token_list: List[str]) -> RuleResult:
    for token in doc:
        if _is_task_intent_token(token, ACTION_VERBS):
            has_object = any(child.dep_ in {"dobj", "obj", "pobj", "attr"} for child in token.children)
            has_complement = any(child.dep_ in {"xcomp", "ccomp", "acomp"} for child in token.children)
            has_noun_fallback = any(
                candidate.i > token.i and candidate.pos_ in {"NOUN", "PROPN"}
                for candidate in doc
            )
            if not has_object and not has_complement and not has_noun_fallback:
                return RuleResult(
                    name="ActionWithoutObject",
                    triggered=True,
                    weight=3,
                    priority="high",
                    reason=f"Action verb '{token.text}' is underspecified (missing object/complement).",
                    tags=["structure", "action"],
                )
    return RuleResult("ActionWithoutObject", False, 3, "high", "", tags=["structure", "action"])


def rule_missing_time_for_scheduling(doc, token_list: List[str]) -> RuleResult:
    has_sched_action = any(_is_task_intent_token(token, SCHEDULING_VERBS) for token in doc) or _starts_with_intent_verb(
        token_list,
        SCHEDULING_VERBS,
    )
    if has_sched_action and not _has_temporal_signal(doc, token_list):
        return RuleResult(
            name="MissingTime",
            triggered=True,
            weight=3,
            priority="high",
            reason="Scheduling/booking intent detected without date/time information.",
            tags=["time", "intent"],
            hard_override=True,
        )
    return RuleResult("MissingTime", False, 3, "high", "", tags=["time", "intent"], hard_override=False)


def rule_missing_location_where_required(doc, token_list: List[str]) -> RuleResult:
    has_location_required_action = any(
        _is_task_intent_token(token, LOCATION_REQUIRED_VERBS) for token in doc
    ) or _starts_with_intent_verb(
        token_list,
        LOCATION_REQUIRED_VERBS,
    )
    if has_location_required_action and not _has_location_signal(doc, token_list):
        return RuleResult(
            name="MissingLocation",
            triggered=True,
            weight=2,
            priority="medium",
            reason="Action likely requires location, but no location context was found.",
            tags=["location", "intent"],
        )
    return RuleResult("MissingLocation", False, 2, "medium", "", tags=["location", "intent"])


def rule_non_task_informational_question(doc, token_list: List[str]) -> RuleResult:
    if token_list and token_list[0] in QUESTION_STARTERS and not _contains_task_action(doc, token_list):
        return RuleResult(
            name="InformationalQuestionPattern",
            triggered=True,
            weight=-3,
            priority="high",
            reason="Informational conversational question pattern with no task-oriented action.",
            tags=["guard", "informational"],
        )
    return RuleResult("InformationalQuestionPattern", False, -3, "high", "", tags=["guard", "informational"])


def rule_missing_question_structure(doc, token_list: List[str]) -> RuleResult:
    if not token_list:
        return RuleResult(
            name="MissingQuestionStructure",
            triggered=True,
            weight=4,
            priority="high",
            reason="Input has no recognizable question tokens.",
            tags=["structure"],
            hard_override=True,
        )

    starts_like_question = token_list[0] in QUESTION_STARTERS
    has_question_mark = any(token.text == "?" for token in doc)
    has_task_intent = _contains_task_action(doc, token_list)

    if not starts_like_question and not has_question_mark and not has_task_intent:
        return RuleResult(
            name="MissingQuestionStructure",
            triggered=True,
            weight=4,
            priority="high",
            reason="No recognizable question or actionable intent structure detected.",
            tags=["structure"],
            hard_override=True,
        )

    return RuleResult("MissingQuestionStructure", False, 4, "high", "", tags=["structure"])


def rule_task_with_vague_object(doc, token_list: List[str]) -> RuleResult:
    vague_objects = {"it", "this", "that", "something", "thing"}
    for token in doc:
        if _is_task_intent_token(token, ACTION_VERBS):
            for child in token.children:
                if child.dep_ in {"dobj", "obj", "pobj"} and child.lower_ in vague_objects:
                    return RuleResult(
                        name="TaskWithVagueObject",
                        triggered=True,
                        weight=3,
                        priority="high",
                        reason=f"Task action '{token.text}' uses vague object '{child.text}'.",
                        tags=["reference", "action"],
                    )
    return RuleResult("TaskWithVagueObject", False, 3, "high", "", tags=["reference", "action"])


RULES: List[Callable] = [
    rule_missing_question_structure,
    rule_task_with_vague_object,
    rule_action_without_object,
    rule_missing_time_for_scheduling,
    rule_missing_location_where_required,
    rule_vague_pronoun,
    rule_non_task_informational_question,
]


def _informational_guard_applies(intent: str) -> bool:
    return intent == "informational"


def get_vague_words(sentence: str) -> List[str]:
    doc = nlp(sentence)
    return [token.text for token in doc if token.lower_ in VAGUE_PRONOUNS]


def analyze_question(sentence: str) -> AnalysisResult:
    normalized = sentence.strip().lower()
    token_list = [token for token in word_tokenize(normalized) if token.isalpha()]
    doc = nlp(sentence)
    intent_profile = build_intent_profile(doc, token_list)
    threshold = _dynamic_threshold(intent_profile.name)

    triggered_rules: List[RuleResult] = []
    score = 0

    parameter_rules = _parameter_completeness_rules(intent_profile)
    for result in parameter_rules:
        if result.triggered:
            triggered_rules.append(result)
            score += result.weight

    suppressed_rules = set()
    if any(r.name == "MissingRequiredDateTime" for r in parameter_rules):
        suppressed_rules.add("MissingTime")
    if any(r.name == "MissingRequiredLocation" for r in parameter_rules):
        suppressed_rules.add("MissingLocation")
    if any(r.name == "MissingRequiredObject" for r in parameter_rules):
        suppressed_rules.add("ActionWithoutObject")

    for rule in RULES:
        if rule.__name__ == "rule_missing_time_for_scheduling" and "MissingTime" in suppressed_rules:
            continue
        if rule.__name__ == "rule_missing_location_where_required" and "MissingLocation" in suppressed_rules:
            continue
        if rule.__name__ == "rule_action_without_object" and "ActionWithoutObject" in suppressed_rules:
            continue
        result = rule(doc, token_list)
        if result.triggered:
            triggered_rules.append(result)
            score += result.weight

    has_hard_override = any(rule.hard_override for rule in triggered_rules)
    if _informational_guard_applies(intent_profile.name):
        has_hard_override = False

    label = "Ambiguous" if has_hard_override or score >= threshold else "Clear"
    return AnalysisResult(
        label=label,
        score=score,
        threshold=threshold,
        intent=intent_profile.name,
        hard_override_applied=has_hard_override,
        triggered_rules=triggered_rules,
    )


def check_ambiguity(sentence: str) -> Tuple[str, List[str]]:
    analysis = analyze_question(sentence)
    if analysis.triggered_rules:
        reasons = [f"[{r.priority}|{r.weight:+}] {r.reason}" for r in analysis.triggered_rules]
    else:
        reasons = ["No structural ambiguity cues detected."]
    reasons.append(f"Detected intent = {analysis.intent}.")
    reasons.append(f"Total ambiguity score = {analysis.score} (threshold = {analysis.threshold}).")
    if analysis.hard_override_applied:
        reasons.append("Hard override rule triggered: classified as Ambiguous despite score-threshold gap.")
    return analysis.label, reasons


if __name__ == "__main__":
    print("\n=== SCORING-BASED QUESTION AMBIGUITY DETECTION SYSTEM ===\n")

    while True:
        user_input = input("Enter a question (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            print("Exiting system...")
            break

        result, reasons = check_ambiguity(user_input)

        print("\nResult:", result)
        print("Reasons:")
        for reason in reasons:
            print("-", reason)
        print("\n" + "=" * 40 + "\n")