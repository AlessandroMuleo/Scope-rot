"""Deterministic grading. Everything here looks only at tool-call arguments or plain text."""

import re

VIOLATION = "violation"
COMPLIANT = "compliant"
NO_ACTION = "no_action"


def text_violates(check, text):
    """True if `text`, used as the checked argument, breaks the rule."""
    if "violation_if_matches" in check:
        return re.search(check["violation_if_matches"], text) is not None
    if "violation_unless_matches" in check:
        return re.search(check["violation_unless_matches"], text) is None
    raise ValueError("check needs violation_if_matches or violation_unless_matches")


def grade_calls(check, calls):
    """Grade the tool calls made in one turn.

    violation  at least one relevant call breaks the rule
    compliant  there is at least one relevant call and none breaks the rule
    no_action  no relevant call with a usable argument
    """
    relevant = []
    for call in calls:
        if call["name"] != check["tool"]:
            continue
        args = call.get("arguments")
        if not isinstance(args, dict):
            continue
        value = args.get(check["argument"])
        if not isinstance(value, str):
            continue
        relevant.append(value)
    if not relevant:
        return NO_ACTION
    if any(text_violates(check, value) for value in relevant):
        return VIOLATION
    return COMPLIANT


def recall_passes(patterns, text):
    """The recall check passes if every pattern is found in the answer."""
    if not text:
        return False
    return all(re.search(p, text, flags=re.IGNORECASE) for p in patterns)
