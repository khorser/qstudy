"""
AggregatorClient -- duck-types just enough of anthropic.Anthropic's interface
(client.messages.create(model=, max_tokens=, system=, messages=) ->
response.content = [block with .type == "text", .text]) for
ai_narrator.explain_slice() and agentic_analyst.run_agent() to work
unmodified against an OpenAI-compatible aggregator endpoint instead of the
Anthropic API directly -- same seam as ollama_client.OllamaClient.

Talks to POST {base_url}/chat/completions with OpenAI's request/response
shape (messages with "system"/"user"/"assistant" roles in one list, choices
[0].message.content for the reply), not Ollama's or Anthropic's native
format.

Deliberately does not hardcode a base URL -- pass one explicitly or set
AGGREGATOR_BASE_URL, so no third-party service name lives in this file.
Referred to elsewhere in this repo (agentic_testing_notes.md) by the
pseudonym "ModelProxy," not by its real name.

The model catalog behind a given aggregator/model name isn't independently
verified here -- see agentic_testing_notes.md for the trust caveats
surfaced before this was used the first time.
"""
import os
from types import SimpleNamespace

import requests


class _Messages:
    def __init__(self, client):
        self._client = client

    def create(self, model, max_tokens=None, system=None, messages=None, **kwargs):
        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages or [])

        payload = {
            "model": model,
            "messages": payload_messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = requests.post(
            f"{self._client.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._client.api_key}"},
            timeout=self._client.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""

        # Mimic anthropic's Message.content: a list of blocks with .type/.text,
        # since ai_narrator.explain_slice() does:
        #   "".join(block.text for block in response.content if block.type == "text")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class AggregatorClient:
    def __init__(self, base_url=None, api_key=None, timeout=180):
        base_url = base_url or os.environ.get("AGGREGATOR_BASE_URL")
        if not base_url:
            raise ValueError(
                "No base URL -- pass base_url= or set AGGREGATOR_BASE_URL."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key -- pass api_key= or set ANTHROPIC_API_KEY "
                "(the env var name this aggregator key was exported under)."
            )
        self.timeout = timeout
        self.messages = _Messages(self)


def list_aggregator_models(base_url=None, api_key=None, timeout=10):
    """Returns a list of model ids the aggregator reports via its
    OpenAI-compatible GET {base_url}/models endpoint. Raises if the
    aggregator isn't reachable or misconfigured -- callers should catch
    and show a friendly message rather than let this propagate into a
    dead dropdown."""
    base_url = base_url or os.environ.get("AGGREGATOR_BASE_URL")
    if not base_url:
        raise ValueError("No base URL -- pass base_url= or set AGGREGATOR_BASE_URL.")
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key -- pass api_key= or set ANTHROPIC_API_KEY.")
    resp = requests.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    ids = [m.get("id") for m in data.get("data", [])]
    return sorted(i for i in ids if i)
