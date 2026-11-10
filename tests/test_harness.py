"""Tests for the harness. No real model is called here.

    python -m unittest discover -s tests -v
"""

import copy
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from harness import analyze, grading
from harness.backends import MockBackend, OpenAIBackend
from harness.runner import run_all, run_episode
from harness.scenario import CONDITIONS, build_turns, load_families, validate_family

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILIES = load_families(os.path.join(ROOT, "scenarios"))
CASES = [(family, scenario) for family in FAMILIES for scenario in family["scenarios"]]
# The fake HTTP provider below answers with write_log calls, so it needs a logging scenario.
LOGGING_CASE = next(case for case in CASES if case[0]["family"] == "logging")


class ScenarioTests(unittest.TestCase):
    def test_examples_match_the_check(self):
        for _, scenario in CASES:
            check, examples = scenario["check"], scenario["examples"]
            self.assertFalse(grading.text_violates(check, examples["compliant"]), scenario["id"])
            self.assertTrue(grading.text_violates(check, examples["violating"]), scenario["id"])

    def test_rule_text_passes_its_own_recall_check(self):
        for _, scenario in CASES:
            self.assertTrue(grading.recall_passes(scenario["recall_patterns"], scenario["rule"]),
                            scenario["id"])

    def test_conditions_have_the_same_shape(self):
        for _, scenario in CASES:
            counts = {c: len(build_turns(scenario, c)) for c in CONDITIONS}
            self.assertEqual(len(set(counts.values())), 1, scenario["id"])
            for condition in CONDITIONS:
                self.assertEqual(build_turns(scenario, condition)[-1]["kind"], "trigger")

    def test_paired_conditions_have_similar_length(self):
        # The control must not be shorter in a way that could explain a difference.
        for _, scenario in CASES:
            for treated, baseline in analyze.PAIRS.values():
                a = sum(len(t["text"]) for t in build_turns(scenario, treated))
                b = sum(len(t["text"]) for t in build_turns(scenario, baseline))
                self.assertLess(abs(a - b) / b, 0.10, f"{scenario['id']} {treated}: {a} vs {b}")

    def test_exception_end_does_not_repeat_the_rule(self):
        for _, scenario in CASES:
            end = scenario["turns"]["exception_end"]
            self.assertFalse(grading.recall_passes(scenario["recall_patterns"], end), scenario["id"])

    def test_lookup_turns_name_an_id(self):
        # The mock calls the family lookup tool with the id found in the turn text.
        for family, scenario in CASES:
            pattern = family["lookup"]["id_pattern"]
            turns = scenario["turns"]
            for text in turns["during_filler"] + turns["gap"] + [turns["hold"]]:
                self.assertRegex(text, pattern, scenario["id"])
            for text in turns["warmup"] + turns["during_action"] + [turns["trigger"]]:
                self.assertNotRegex(text, pattern, scenario["id"])

    def test_bad_lookup_is_rejected(self):
        family = copy.deepcopy(FAMILIES[0])
        family["lookup"]["tool"] = "no_such_tool"
        with self.assertRaises(ValueError):
            validate_family(family)
        family = copy.deepcopy(FAMILIES[0])
        family["lookup"]["argument"] = "no_such_argument"
        with self.assertRaises(ValueError):
            validate_family(family)
        family = copy.deepcopy(FAMILIES[0])
        family["lookup"]["tool"] = family["scenarios"][0]["check"]["tool"]
        family["lookup"]["argument"] = family["scenarios"][0]["check"]["argument"]
        with self.assertRaises(ValueError):
            validate_family(family)
        family = copy.deepcopy(FAMILIES[0])
        del family["lookup"]
        validate_family(family)

    def test_true_phase_per_condition(self):
        _, scenario = CASES[0]
        phase = {c: build_turns(scenario, c)[-1]["phase"] for c in CONDITIONS}
        self.assertEqual(phase["exception_active"], "exception")
        for condition in ("control", "reactivate", "control_noact", "reactivate_noact"):
            self.assertEqual(phase[condition], "rule")
        kinds = [t["kind"] for t in build_turns(scenario, "reactivate_noact")]
        self.assertNotIn("action", kinds)


class GradingTests(unittest.TestCase):
    CHECK = {"tool": "write_log", "argument": "message", "violation_if_matches": r"\d{10}"}

    def test_outcomes(self):
        ok = {"name": "write_log", "arguments": {"message": "Account ****1234"}}
        bad = {"name": "write_log", "arguments": {"message": "Account 1234567890"}}
        other = {"name": "get_ticket", "arguments": {"ticket_id": "1234567890"}}
        broken = {"name": "write_log", "arguments": None}
        self.assertEqual(grading.grade_calls(self.CHECK, [ok]), grading.COMPLIANT)
        self.assertEqual(grading.grade_calls(self.CHECK, [ok, bad]), grading.VIOLATION)
        self.assertEqual(grading.grade_calls(self.CHECK, [other]), grading.NO_ACTION)
        self.assertEqual(grading.grade_calls(self.CHECK, [broken]), grading.NO_ACTION)
        self.assertEqual(grading.grade_calls(self.CHECK, []), grading.NO_ACTION)


class MockRunTests(unittest.TestCase):
    def outcomes(self, policy):
        backend = MockBackend(policy)
        result = {}
        for family, scenario in CASES:
            for condition in CONDITIONS:
                record = run_episode(backend, family, scenario, condition, rep=0)
                self.assertIsNone(record["error"])
                result[(scenario["id"], condition)] = record
        return result

    def test_perfect_agent(self):
        for (_, condition), record in self.outcomes("perfect").items():
            expected = grading.VIOLATION if condition == "exception_active" else grading.COMPLIANT
            self.assertEqual(record["trigger_outcome"], expected, condition)
            self.assertTrue(record["recall_pass"])

    def test_sticky_agent_fails_only_after_an_exception(self):
        for (_, condition), record in self.outcomes("sticky").items():
            expected = grading.COMPLIANT if condition.startswith("control") else grading.VIOLATION
            self.assertEqual(record["trigger_outcome"], expected, condition)

    def test_amnesiac_fails_recall(self):
        for record in self.outcomes("amnesiac").values():
            self.assertFalse(record["recall_pass"])

    def test_mock_uses_the_family_lookup_tool(self):
        for family, scenario in CASES:
            record = run_episode(MockBackend("perfect"), family, scenario, "control_noact", rep=0)
            lookup = family["lookup"]
            for turn in record["turns"]:
                if turn["kind"] not in ("filler", "gap"):
                    continue
                self.assertEqual(len(turn["calls"]), 1, scenario["id"])
                call = turn["calls"][0]
                self.assertEqual(call["name"], lookup["tool"])
                self.assertRegex(call["arguments"][lookup["argument"]], f"^{lookup['id_pattern']}$")
            results = [m["content"] for m in record["messages"] if m["role"] == "tool"]
            self.assertFalse(any(r.startswith("error") for r in results), scenario["id"])

    def test_mock_without_lookup_answers_in_text(self):
        family, scenario = CASES[0]
        family = {k: v for k, v in family.items() if k != "lookup"}
        record = run_episode(MockBackend("sticky"), family, scenario, "reactivate_noact", rep=0)
        self.assertIsNone(record["error"])
        for turn in record["turns"]:
            if turn["kind"] in ("filler", "gap"):
                self.assertEqual(turn["calls"], [])
        self.assertEqual(record["trigger_outcome"], grading.VIOLATION)

    def test_transcript_is_well_formed(self):
        family, scenario = CASES[0]
        record = run_episode(MockBackend("perfect"), family, scenario, "reactivate", rep=0)
        messages = record["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(scenario["rule"].strip(), messages[0]["content"])
        open_ids = set()
        for message in messages:
            for call in message.get("tool_calls", []):
                open_ids.add(call["id"])
            if message["role"] == "tool":
                self.assertIn(message["tool_call_id"], open_ids)
        users = [m for m in messages if m["role"] == "user"]
        self.assertEqual(len(users), len(build_turns(scenario, "reactivate")) + 1)


class AnalysisTests(unittest.TestCase):
    def run_policy(self, policy, reps=3):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "run.jsonl")
            log = open(os.devnull, "w")
            written = run_all(MockBackend(policy), FAMILIES, path, reps, log=log)
            self.assertEqual(written, len(CASES) * len(CONDITIONS) * reps)
            # Running again must not duplicate finished episodes.
            self.assertEqual(run_all(MockBackend(policy), FAMILIES, path, reps, log=log), 0)
            log.close()
            return analyze.load([path])

    def test_deltas(self):
        for policy, expected in (("perfect", 0.0), ("sticky", 1.0), ("always_violate", 0.0)):
            rows = self.run_policy(policy)
            entry = analyze.summarise(rows, iterations=200)[f"mock-{policy}"]
            for name in ("delta_scope", "delta_scope_noact"):
                point, low, high = entry[name]
                self.assertAlmostEqual(point, expected, msg=f"{policy} {name}")
                self.assertLessEqual(low, point)
                self.assertGreaterEqual(high, point)
            self.assertAlmostEqual(analyze.cross_model_delta(rows, iterations=200)[0], expected)

    def test_recall_filter_and_adoption(self):
        entry = analyze.summarise(self.run_policy("amnesiac"), iterations=50)["mock-amnesiac"]
        self.assertAlmostEqual(entry["delta_scope"][0], 1.0)
        self.assertIsNone(entry["delta_scope_recall_only"][0])
        self.assertAlmostEqual(entry["exception_adoption"], 1.0)
        entry = analyze.summarise(self.run_policy("always_comply"), iterations=50)
        self.assertAlmostEqual(entry["mock-always_comply"]["exception_adoption"], 0.0)
        self.assertIn("delta_scope", analyze.report(entry))

    def test_model_that_ignores_the_exception_is_left_out(self):
        comply = self.run_policy("always_comply")
        self.assertTrue(analyze.summarise(comply, iterations=50)["mock-always_comply"]["excluded"])
        self.assertIsNone(analyze.cross_model_delta(comply, iterations=50)[0])
        sticky = self.run_policy("sticky")
        self.assertFalse(analyze.summarise(sticky, iterations=50)["mock-sticky"]["excluded"])
        # The excluded model does not pull the mean down.
        self.assertAlmostEqual(analyze.cross_model_delta(comply + sticky, iterations=50)[0], 1.0)


class _FakeProvider(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server: tool call first, then a text answer."""

    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.headers.get("Authorization"), body))
        last = body["messages"][-1]
        if last["role"] == "tool":
            message = {"role": "assistant", "content": "Logged."}
        elif "standing rule" in last["content"]:
            message = {"role": "assistant", "content": "Always redact email addresses."}
        else:
            # Arguments as an object, as some local servers send them.
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "function": {"name": "write_log",
                             "arguments": {"message": "User [REDACTED] had a problem."}}}]}
        payload = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class OpenAIBackendTests(unittest.TestCase):
    def test_episode_over_http(self):
        _FakeProvider.requests = []
        server = HTTPServer(("127.0.0.1", 0), _FakeProvider)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = OpenAIBackend(f"http://127.0.0.1:{server.server_port}/v1", "fake-model",
                                    api_key="k", temperature=0.5, retries=0)
            family, scenario = LOGGING_CASE
            record = run_episode(backend, family, scenario, "control", rep=0, seed=7)
        finally:
            server.shutdown()
            server.server_close()
        self.assertIsNone(record["error"])
        self.assertEqual(record["trigger_outcome"], grading.COMPLIANT)
        self.assertTrue(record["recall_pass"])
        auth, body = _FakeProvider.requests[0]
        self.assertEqual(auth, "Bearer k")
        self.assertEqual(body["model"], "fake-model")
        self.assertEqual(body["seed"], 7)
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["tools"][0]["function"]["name"], "write_log")
        tool_message = next(m for _, b in _FakeProvider.requests for m in b["messages"]
                            if m["role"] == "tool")
        self.assertTrue(tool_message["tool_call_id"])

    def test_backend_error_is_recorded(self):
        backend = OpenAIBackend("http://127.0.0.1:9/v1", "fake-model", retries=0, timeout=2)
        family, scenario = CASES[0]
        record = run_episode(backend, family, scenario, "control", rep=0)
        self.assertIsNotNone(record["error"])
        self.assertIsNone(record["trigger_outcome"])


if __name__ == "__main__":
    unittest.main()
