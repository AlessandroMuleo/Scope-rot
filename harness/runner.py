"""Runs episodes and writes one JSON line per episode.

Example with a local Ollama model:

    python -m harness.runner --base-url http://localhost:11434/v1 \
        --model qwen3:8b --reps 20 --out results/qwen3-8b.jsonl

Hosted provider (the key is read from the SCOPE_ROT_API_KEY environment variable):

    python -m harness.runner --base-url https://openrouter.ai/api/v1 \
        --model <model id> --reps 20 --out results/<name>.jsonl
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

from . import grading
from .backends import MOCK_POLICIES, BackendError, MockBackend, OpenAIBackend
from .scenario import CONDITIONS, build_turns, load_families, system_prompt, tool_specs

GRADED_KINDS = ("warmup", "action", "trigger")


def _parse_arguments(raw):
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _run_turn(backend, messages, tools, family, hints, seed, max_steps):
    """Send one user turn, answer tool calls with simulated results, return the calls."""
    calls = []
    final_text = ""
    for _ in range(max_steps):
        reply = backend.chat(messages, tools, hints=hints, seed=seed)
        messages.append(reply)
        final_text = reply.get("content") or ""
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            break
        for call in tool_calls:
            name = call["function"]["name"]
            raw = call["function"]["arguments"]
            calls.append({"name": name, "arguments": _parse_arguments(raw), "raw": raw})
            result = family["tool_results"].get(name, f"error: unknown tool {name}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return calls, final_text


def run_episode(backend, family, scenario, condition, rep, seed=None, max_steps=4):
    tools = tool_specs(family)
    check = scenario["check"]
    messages = [{"role": "system", "content": system_prompt(family, scenario)}]
    record = {
        "family": family["family"],
        "scenario": scenario["id"],
        "condition": condition,
        "rep": rep,
        "seed": seed,
        "turns": [],
        "trigger_outcome": None,
        "recall_pass": None,
        "error": None,
    }
    exception_seen = False
    try:
        for turn in build_turns(scenario, condition):
            if turn["kind"] == "exception_start":
                exception_seen = True
            hints = {
                "kind": turn["kind"],
                "phase": turn["phase"],
                "exception_seen": exception_seen,
                "check": check,
                "examples": scenario["examples"],
                "rule": scenario["rule"],
            }
            messages.append({"role": "user", "content": turn["text"]})
            calls, _ = _run_turn(backend, messages, tools, family, hints, seed, max_steps)
            entry = {"kind": turn["kind"], "phase": turn["phase"], "calls": calls}
            if turn["kind"] in GRADED_KINDS:
                entry["outcome"] = grading.grade_calls(check, calls)
            record["turns"].append(entry)
            if turn["kind"] == "trigger":
                record["trigger_outcome"] = entry["outcome"]

        # Recall check, after the trigger so it cannot influence it.
        messages.append({"role": "user", "content": family["recall_probe"].strip()})
        hints = {"kind": "recall", "rule": scenario["rule"]}
        _, answer = _run_turn(backend, messages, tools, family, hints, seed, max_steps=1)
        record["recall_answer"] = answer
        record["recall_pass"] = grading.recall_passes(scenario["recall_patterns"], answer)
    except BackendError as error:
        record["error"] = str(error)
    record["messages"] = messages
    return record


def _git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _done_keys(path):
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error") is None:
                keys.add((row["model"], row["scenario"], row["condition"], row["rep"]))
    return keys


def run_all(backend, families, out_path, reps, conditions=CONDITIONS, seed_base=None,
            scenario_ids=None, log=sys.stderr):
    """Run every (scenario, condition, rep) cell that is not already in out_path."""
    meta = backend.describe()
    meta["harness_commit"] = _git_commit()
    done = _done_keys(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    written = 0
    with open(out_path, "a", encoding="utf-8") as handle:
        for family in families:
            for scenario in family["scenarios"]:
                if scenario_ids and scenario["id"] not in scenario_ids:
                    continue
                for condition in conditions:
                    for rep in range(reps):
                        key = (meta["model"], scenario["id"], condition, rep)
                        if key in done:
                            continue
                        seed = None if seed_base is None else seed_base + rep
                        record = run_episode(backend, family, scenario, condition, rep, seed)
                        record.update(meta)
                        record["timestamp"] = datetime.datetime.now(
                            datetime.timezone.utc).isoformat(timespec="seconds")
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        written += 1
                        status = record["error"] or record["trigger_outcome"]
                        print(f"{scenario['id']} {condition} rep={rep}: {status}", file=log)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Scope-rot episodes.")
    parser.add_argument("--scenarios", default="scenarios", help="directory with the YAML families")
    parser.add_argument("--out", required=True, help="JSONL file to append to")
    parser.add_argument("--reps", type=int, default=20, help="repetitions per cell")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--scenario-ids", nargs="+", help="run only these scenarios")
    parser.add_argument("--backend", choices=("openai", "mock"), default="openai")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed-base", type=int, help="if set, rep r is sent seed-base + r")
    parser.add_argument("--mock-policy", choices=MOCK_POLICIES, default="perfect")
    args = parser.parse_args(argv)

    if args.backend == "mock":
        backend = MockBackend(args.mock_policy)
    else:
        if not args.model:
            parser.error("--model is required with the openai backend")
        backend = OpenAIBackend(args.base_url, args.model,
                                api_key=os.environ.get("SCOPE_ROT_API_KEY"),
                                temperature=args.temperature, max_tokens=args.max_tokens)
    families = load_families(args.scenarios)
    written = run_all(backend, families, args.out, args.reps, args.conditions,
                      args.seed_base, args.scenario_ids)
    print(f"wrote {written} episodes to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
