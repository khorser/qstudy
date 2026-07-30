"""
OllamaClient -- duck-types just enough of anthropic.Anthropic's interface
(client.messages.create(model=, max_tokens=, system=, messages=) ->
response.content = [block with .type == "text", .text]) for
ai_narrator.explain_slice() and AICircuitSlicer to work unmodified against
a local Ollama server instead of the Anthropic API.

Nothing in resource_stats.py, ai_narrator.py, or ai_circuit_slicer.py
changes -- this is purely an alternative "client" object, matching the
existing seam (`anthropic_client=` in AICircuitSlicer.__init__).
"""
import re
from types import SimpleNamespace

import requests

# Reasoning-style local models (DeepSeek-R1, Qwen3-thinking, etc.) often
# embed their chain-of-thought directly in message.content wrapped in
# <think>...</think>, rather than in a separate field. Since this tool's
# job is a short grounded narration (not a task that benefits from visible
# reasoning -- see note in compare_backends/README), we strip it so it
# doesn't pollute the displayed narration or the coverage scorer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# If max_tokens runs out mid-reasoning, the response can contain an opening
# <think> with no closing tag -- observed with phi4-mini-reasoning:3.8b at
# the default max_tokens=300, which never got far enough to finish a single
# slice's reasoning. _THINK_BLOCK's non-greedy match requires a closing tag
# and leaves an unterminated block untouched, which used to leak raw,
# mid-sentence chain-of-thought as if it were the narration.
_THINK_OPEN_UNCLOSED = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text, truncated=False):
    """Strip a complete <think>...</think> block unconditionally.

    An *unterminated* <think> is ambiguous and needs `truncated` (whether
    generation was cut short by the token budget rather than reaching a
    natural stop) to resolve correctly:
    - truncated=True: the budget ran out mid-thought. There's no real
      answer in what's left, just reasoning that never finished, so
      discard from <think> through end of string.
    - truncated=False: generation stopped on its own but still left an
      unterminated <think>. Observed with phi4-mini-reasoning:3.8b, which
      apparently never emits a closing tag at all, even when it's
      genuinely done -- its "reasoning" and its answer aren't separated
      by any reliable delimiter. Discarding here would throw away a
      legitimate answer, so only the bare opening tag(s) are dropped and
      the rest of the text (reasoning blended with the real answer) is
      kept. The same model has also been observed re-opening a second
      bare <think> mid-response (e.g. as a garbled "</> <think>"), so all
      occurrences are stripped, not just the first.
    """
    text = _THINK_BLOCK.sub("", text)
    if "<think>" in text:
        if truncated:
            text = _THINK_OPEN_UNCLOSED.sub("", text)
        else:
            text = text.replace("<think>", "")
    return text.strip()


class _Messages:
    def __init__(self, client):
        self._client = client

    def create(self, model, max_tokens=None, system=None, messages=None,
               think=False, **kwargs):
        """
        think: for reasoning-capable models (Qwen3, DeepSeek-R1, etc.),
        recent Ollama versions accept a native "think" toggle. Defaults to
        False here on purpose: this tool's job is a short grounded
        narration over already-verified facts, not a task that benefits
        from chain-of-thought (see README) -- and leaving thinking on
        risks the model spending the whole max_tokens budget on hidden
        reasoning and never emitting visible content. Pass think=True to
        override if you want to compare.
        """
        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages or [])

        payload = {
            "model": model,
            "messages": payload_messages,
            "stream": False,
            "think": think,
        }
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}

        resp = requests.post(
            f"{self._client.base_url}/api/chat",
            json=payload,
            timeout=self._client.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        # done_reason "length" means num_predict was hit before a natural
        # stop -- see _strip_thinking's truncated= parameter.
        truncated = data.get("done_reason") not in (None, "stop")
        text = _strip_thinking(message.get("content", "") or "", truncated=truncated)

        if not text:
            # Some models/Ollama versions route everything into a separate
            # "thinking" field even with think=False, or the model burned
            # the whole token budget on hidden reasoning before emitting
            # any content. Surface *something* rather than silence, and
            # say why, rather than returning an empty string that looks
            # like a bug in this code instead of a model/budget issue.
            thinking = message.get("thinking", "")
            if thinking:
                text = (
                    "[No visible answer content came back -- only a "
                    "'thinking' field, which usually means max_tokens ran "
                    "out during reasoning, or the model ignored think=False. "
                    "Try a much higher max_tokens, or a non-reasoning model "
                    "tag, for this narration task. Raw thinking below:]\n\n"
                    + thinking
                )
            else:
                text = (
                    "[Empty response from Ollama -- check `ollama ps` / "
                    "server logs; the model may have errored or the "
                    "request timed out before completion.]"
                )
        # Mimic anthropic's Message.content: a list of blocks with .type/.text,
        # since ai_narrator.explain_slice() does:
        #   "".join(block.text for block in response.content if block.type == "text")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", timeout=300):
        # 120s was too short for reasoning-heavy local models at a
        # max_tokens budget large enough to actually finish (observed:
        # phi4-mini-reasoning:3.8b routinely takes 100-200+s per slice at
        # max_tokens=3500). 300s gives headroom without being unbounded.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.messages = _Messages(self)


def list_ollama_models(base_url="http://localhost:11434", timeout=10):
    """Returns a list of model tags installed on the given Ollama server
    (e.g. ["qwen2.5:7b", "deepseek-r1:8b"]). Raises if the server isn't
    reachable -- callers should catch and show a friendly message rather
    than let this propagate into a dead dropdown."""
    resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    names = [m.get("name") or m.get("model") for m in data.get("models", [])]
    return [n for n in names if n]
