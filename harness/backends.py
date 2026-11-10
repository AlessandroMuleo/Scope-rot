"""Model backends.

OpenAIBackend talks to any OpenAI-compatible chat-completions endpoint. That covers
Ollama (http://localhost:11434/v1), OpenRouter, and most hosted providers.

MockBackend is a scripted fake model. It exists to test the harness, never to
produce results.
"""

import json
import re
import time
import urllib.error
import urllib.request


class BackendError(RuntimeError):
    pass


class OpenAIBackend:
    def __init__(self, base_url, model, api_key=None, temperature=1.0,
                 max_tokens=512, timeout=120, retries=4):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def describe(self):
        return {"backend": "openai", "url": self.url, "model": self.model,
                "temperature": self.temperature, "max_tokens": self.max_tokens}

    def chat(self, messages, tools, hints=None, seed=None):
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode("utf-8")

        last_error = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return self._normalise(payload)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {error.code}: {detail}"
                if error.code not in (408, 409, 429) and error.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = repr(error)
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 30))
        raise BackendError(last_error)

    @staticmethod
    def _normalise(payload):
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise BackendError(f"unexpected response: {json.dumps(payload)[:500]}")
        out = {"role": "assistant", "content": message.get("content") or ""}
        calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments if arguments is not None else {})
            calls.append({
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {"name": function.get("name") or "", "arguments": arguments},
            })
        if calls:
            out["tool_calls"] = calls
        return out


MOCK_POLICIES = ("perfect", "sticky", "always_comply", "always_violate", "amnesiac")


class MockBackend:
    """Scripted agent, for tests.

    perfect         follows the rule exactly when it is in force
    sticky          stops following the rule for good once an exception has started
    always_comply   always applies the rule, even under the exception
    always_violate  never applies the rule
    amnesiac        like sticky, and cannot state the rule at the recall check
    """

    def __init__(self, policy="perfect"):
        if policy not in MOCK_POLICIES:
            raise ValueError(f"unknown mock policy: {policy}")
        self.policy = policy
        self.model = f"mock-{policy}"

    def describe(self):
        return {"backend": "mock", "model": self.model}

    def _complies(self, hints):
        if self.policy == "perfect":
            return hints["phase"] == "rule"
        if self.policy in ("sticky", "amnesiac"):
            return not hints["exception_seen"]
        return self.policy == "always_comply"

    def chat(self, messages, tools, hints=None, seed=None):
        if messages[-1]["role"] == "tool":
            return {"role": "assistant", "content": "Done."}
        kind = hints["kind"]
        if kind == "recall":
            text = "I am not sure." if self.policy == "amnesiac" else hints["rule"]
            return {"role": "assistant", "content": text}
        if kind in ("warmup", "action", "trigger"):
            examples = hints["examples"]
            text = examples["compliant"] if self._complies(hints) else examples["violating"]
            check = hints["check"]
            call = {
                "id": f"call_{len(messages)}",
                "type": "function",
                "function": {
                    "name": check["tool"],
                    "arguments": json.dumps({check["argument"]: text}),
                },
            }
            return {"role": "assistant", "content": "", "tool_calls": [call]}
        if kind in ("filler", "gap", "hold"):
            found = re.search(r"T-\d+", messages[-1]["content"])
            call = {
                "id": f"call_{len(messages)}",
                "type": "function",
                "function": {
                    "name": "get_ticket",
                    "arguments": json.dumps({"ticket_id": found.group(0) if found else "T-0"}),
                },
            }
            return {"role": "assistant", "content": "", "tool_calls": [call]}
        return {"role": "assistant", "content": "Understood."}
