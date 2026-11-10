"""Loads scenario families and builds the turn list for each condition."""

import glob
import os

import yaml

# Every condition has the same number of user turns.
#
#   control           rule, neutral start, rule-relevant actions, neutral end, trigger
#   reactivate        rule, exception starts, same actions, exception ends, trigger
#   control_noact     like control, but the middle turns are unrelated lookups
#   reactivate_noact  like reactivate, but the agent never acts under the exception
#   exception_active  exception starts and is still on at the trigger (manipulation check)
CONDITIONS = [
    "control",
    "reactivate",
    "control_noact",
    "reactivate_noact",
    "exception_active",
]

REQUIRED_TURN_KEYS = [
    "warmup",
    "exception_start",
    "neutral_start",
    "during_action",
    "during_filler",
    "hold",
    "exception_end",
    "neutral_end",
    "gap",
    "trigger",
]


def _turn(kind, text, phase):
    # phase is the true state of the rule when the turn is sent: "rule" or "exception".
    return {"kind": kind, "text": text.strip(), "phase": phase}


def build_turns(scenario, condition):
    t = scenario["turns"]
    has_exception = condition in ("reactivate", "reactivate_noact", "exception_active")
    acts_in_middle = condition in ("control", "reactivate", "exception_active")
    exception_ends = condition in ("reactivate", "reactivate_noact")
    middle_phase = "exception" if has_exception else "rule"
    final_phase = "exception" if condition == "exception_active" else "rule"

    turns = [_turn("warmup", text, "rule") for text in t["warmup"]]

    if has_exception:
        turns.append(_turn("exception_start", t["exception_start"], "exception"))
    else:
        turns.append(_turn("neutral_start", t["neutral_start"], "rule"))

    if acts_in_middle:
        turns += [_turn("action", text, middle_phase) for text in t["during_action"]]
    else:
        turns += [_turn("filler", text, middle_phase) for text in t["during_filler"]]

    if exception_ends:
        turns.append(_turn("exception_end", t["exception_end"], "rule"))
    elif condition == "exception_active":
        turns.append(_turn("hold", t["hold"], "exception"))
    else:
        turns.append(_turn("neutral_end", t["neutral_end"], "rule"))

    turns += [_turn("gap", text, final_phase) for text in t["gap"]]
    turns.append(_turn("trigger", t["trigger"], final_phase))
    return turns


def system_prompt(family, scenario):
    return f"{family['preamble'].strip()}\n\nStanding rule: {scenario['rule'].strip()}"


def tool_specs(family):
    """Tool definitions in the OpenAI chat-completions format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in family["tools"]
    ]


def validate_family(family, path="<family>"):
    for key in ("family", "preamble", "tools", "tool_results", "recall_probe", "scenarios"):
        if key not in family:
            raise ValueError(f"{path}: missing '{key}'")
    tool_names = {tool["name"] for tool in family["tools"]}
    seen = set()
    for scenario in family["scenarios"]:
        sid = scenario.get("id", "<no id>")
        where = f"{path}: {sid}"
        if sid in seen:
            raise ValueError(f"{where}: duplicate id")
        seen.add(sid)
        for key in ("rule", "check", "examples", "recall_patterns", "turns"):
            if key not in scenario:
                raise ValueError(f"{where}: missing '{key}'")
        check = scenario["check"]
        if check.get("tool") not in tool_names:
            raise ValueError(f"{where}: check.tool is not one of the family tools")
        kinds = [k for k in ("violation_if_matches", "violation_unless_matches") if k in check]
        if len(kinds) != 1 or "argument" not in check:
            raise ValueError(f"{where}: check needs 'argument' and exactly one violation_* key")
        for key in REQUIRED_TURN_KEYS:
            if key not in scenario["turns"]:
                raise ValueError(f"{where}: turns is missing '{key}'")
        if len(scenario["turns"]["during_action"]) != len(scenario["turns"]["during_filler"]):
            raise ValueError(f"{where}: during_action and during_filler must have the same length")
        lengths = {len(build_turns(scenario, c)) for c in CONDITIONS}
        if len(lengths) != 1:
            raise ValueError(f"{where}: conditions have different numbers of turns")


def load_families(directory):
    families = []
    for path in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
        with open(path, encoding="utf-8") as handle:
            family = yaml.safe_load(handle)
        validate_family(family, path)
        families.append(family)
    if not families:
        raise FileNotFoundError(f"no scenario files in {directory}")
    return families
