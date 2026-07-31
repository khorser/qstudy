"""
Agentic circuit analyst.

Unlike ai_narrator.py (a fixed prompt over facts already computed for one
slice), this hands the model a small set of tools and an open-ended
question, and lets it decide what to simulate/inspect and in what order.
This is the genuine multi-step planning case: the "right" sequence of tool
calls depends on the question, not on a template. It's also the one place
in this portfolio piece where a reasoning model is worth benchmarking
against a standard one (see compare_agent_backends.py) -- narration
doesn't need chain-of-thought, but deciding "simulate oracle A, then B,
then compare" arguably does.

Deliberately narrow tool surface: the model orchestrates a handful of
known-safe analysis primitives (list_oracles / run_circuit /
get_slice_facts / compare_resource_cost). It never gets arbitrary code
execution -- "agentic" here means tool-call planning, not an open shell.
"""
import json

from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit

from headless_facts import compute_all_slice_facts
from ollama_client import _strip_thinking


# Same DeutschJozsa as elsewhere in this portfolio, extended with a couple
# more oracle variants so there's something nontrivial to compare.
class DeutschJozsa:
    """Determine whether a boolean oracle f: {0,1}^n -> {0,1} is constant or balanced using a single query."""

    def __init__(self, n):
        self.n = n

    def f_0(self):
        return QuantumCircuit(self.n + 1)

    def f_xor(self):
        c = self.f_0()
        c.cx(list(range(self.n)), self.n)
        return c

    def f_not0th(self):
        c = self.f_0()
        c.cx(0, self.n)
        c.x(self.n)
        return c

    def f_i0n1(self):
        c = self.f_0()
        c.cx([0, 1], self.n)
        c.x(self.n)
        return c

    def get_options(self):
        return [("0", self.f_0), ("xor", self.f_xor), ("0:not", self.f_not0th), ("i0n1", self.f_i0n1)]

    def get_circuit(self, f, label=""):
        x = QuantumRegister(self.n, name="x")
        y = AncillaRegister(1, name="y")
        r = ClassicalRegister(self.n, name="r")
        c = QuantumCircuit(x, y, r)
        c.x(y)
        c.barrier(label="init")
        c.h(x)
        c.h(y)
        c.barrier(label="prep")
        c.append(f().to_gate(label=f"$U_{{{label}}}$"), [*x, *y])
        c.barrier(label="apply")
        c.h(x)
        c.barrier(label="done")
        c.measure(x, r)
        return c


class CircuitAnalystTools:
    """Implements each tool as a plain method; run_circuit actually
    triggers a fresh simulation (via headless_facts), it isn't just
    serving pre-computed data -- so which oracles get simulated, and in
    what order, is a real decision the model makes, not a fixed replay."""

    def __init__(self, algo):
        self.algo = algo
        self._oracle_map = dict(algo.get_options())
        self._runs = {}  # oracle name -> {label: SliceFacts}

    def list_oracles(self):
        return {"oracles": list(self._oracle_map.keys())}

    def run_circuit(self, oracle):
        if oracle not in self._oracle_map:
            return {"error": f"unknown oracle '{oracle}', call list_oracles first"}
        f = self._oracle_map[oracle]
        qc = self.algo.get_circuit(f, oracle)
        facts = compute_all_slice_facts(qc, type(self.algo).__name__, self.algo.__doc__ or "")
        self._runs[oracle] = facts
        return {
            "oracle": oracle,
            "slices": list(facts.keys()),
            "note": "Circuit simulated. Use get_slice_facts to inspect a specific slice.",
        }

    def get_slice_facts(self, oracle, label):
        if oracle not in self._runs:
            return {"error": f"oracle '{oracle}' not yet simulated -- call run_circuit first"}
        if label not in self._runs[oracle]:
            return {"error": f"unknown slice '{label}' for oracle '{oracle}'"}
        f = self._runs[oracle][label]
        return {
            "gate_counts": f.gate_counts,
            "entangling_gates": f.entangling_gates,
            "depth": f.depth,
            "measurements": f.measurements,
            "purity": f.purity,
            "entangled_qubits": f.entangled_qubits,
            "pairwise_entanglement": {f"{i}-{j}": e for (i, j), e in f.pairwise_entanglement.items()},
            "cumulative_total_gates": f.cumulative_total_gates,
        }

    def compare_resource_cost(self, oracle_a, oracle_b):
        for o in (oracle_a, oracle_b):
            if o not in self._runs:
                return {"error": f"oracle '{o}' not yet simulated -- call run_circuit first"}
        rows = {}
        for label, a in self._runs[oracle_a].items():
            b = self._runs[oracle_b].get(label)
            if b is None:
                continue
            rows[label] = {
                "entangling_gates": {oracle_a: a.entangling_gates, oracle_b: b.entangling_gates},
                "total_gates_this_slice": {
                    oracle_a: sum(a.gate_counts.values()),
                    oracle_b: sum(b.gate_counts.values()),
                },
            }
        return rows


TOOL_SCHEMAS = [
    {
        "name": "list_oracles",
        "description": "List the oracle names available for this algorithm instance.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_circuit",
        "description": (
            "Build and simulate the circuit for a given oracle name, split "
            "into the same barrier-delimited slices CircuitSlicer uses. "
            "Must be called before get_slice_facts or compare_resource_cost "
            "can be used for that oracle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"oracle": {"type": "string"}},
            "required": ["oracle"],
        },
    },
    {
        "name": "get_slice_facts",
        "description": (
            "Get gate counts, depth, entangling-gate count, purity, and "
            "entanglement facts for one slice of an already-simulated oracle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "oracle": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["oracle", "label"],
        },
    },
    {
        "name": "compare_resource_cost",
        "description": "Compare per-slice gate/entangling-gate counts between two already-simulated oracles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "oracle_a": {"type": "string"},
                "oracle_b": {"type": "string"},
            },
            "required": ["oracle_a", "oracle_b"],
        },
    },
]

AGENT_SYSTEM_PROMPT = (
    "You are a quantum circuit resource-estimation analyst. You have tools "
    "to simulate a circuit for a chosen oracle and inspect per-slice facts "
    "(gate counts, entanglement, purity). You do NOT have access to the raw "
    "circuit or any other source of truth -- ground every claim in tool "
    "results, call run_circuit before inspecting an oracle you haven't "
    "simulated yet, and say so explicitly if a question can't be answered "
    "from the tools available. Be concise in your final answer.\n\n"
    "You must call at least one tool before writing a final answer -- you "
    "have no other source of information about this specific circuit, and "
    "anything you recall about Deutsch-Jozsa in general will not match the "
    "actual oracle definitions or gate structure used here. If you are "
    "about to answer without having called any tool yet in this "
    "conversation, call a tool instead."
)


def run_agent(client, algo, question, model="claude-sonnet-5", max_steps=8, verbose=True,
              max_tokens=1024, temperature=None):
    """Runs the tool-use loop until the model stops calling tools or
    max_steps is hit. Returns (final_text, trace), where trace is the
    ordered list of (tool_name, tool_input, tool_result) actually executed
    -- useful to display, and to diff against a different model/backend's
    trace for the same question."""
    tools = CircuitAnalystTools(algo)
    dispatch = {
        "list_oracles": lambda **kw: tools.list_oracles(),
        "run_circuit": lambda **kw: tools.run_circuit(**kw),
        "get_slice_facts": lambda **kw: tools.get_slice_facts(**kw),
        "compare_resource_cost": lambda **kw: tools.compare_resource_cost(**kw),
    }
    extra = {}
    if temperature is not None:
        extra["temperature"] = temperature

    messages = [{"role": "user", "content": question}]
    trace = []

    for _ in range(max_steps):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=AGENT_SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
            **extra,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return final_text, trace

        tool_results = []
        for block in tool_use_blocks:
            fn = dispatch.get(block.name)
            if fn is None:
                result = {"error": f"unknown tool {block.name}"}
            else:
                try:
                    result = fn(**block.input)
                except Exception as e:
                    result = {"error": str(e)}
            trace.append((block.name, block.input, result))
            if verbose:
                print(f"[tool call] {block.name}({block.input}) -> {json.dumps(result)[:200]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return "[max_steps reached without a final answer]", trace


def _make_dispatch(tools):
    return {
        "list_oracles": lambda **kw: tools.list_oracles(),
        "run_circuit": lambda **kw: tools.run_circuit(**kw),
        "get_slice_facts": lambda **kw: tools.get_slice_facts(**kw),
        "compare_resource_cost": lambda **kw: tools.compare_resource_cost(**kw),
    }


def _tool_schemas_to_ollama(schemas):
    """Anthropic-style {name, description, input_schema} -> Ollama/OpenAI-
    style {"type": "function", "function": {name, description, parameters}}."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


def run_agent_ollama(base_url, algo, question, model, max_steps=8, verbose=True, timeout=120,
                      strip_thinking=True, think=False, max_tokens=1024, temperature=None):
    """Ollama-native equivalent of run_agent(). Run live against a real
    Ollama server many times since this was first written (see
    agentic_testing_notes.md) -- the request/response shape here follows
    Ollama's documented tool-calling format (tools passed OpenAI-style;
    responses carry message["tool_calls"] as a list of
    {"function": {"name", "arguments"}}, with no per-call id the way
    Anthropic/OpenAI have). Tool-calling reliability varies a lot by model
    -- many small local models will just answer in prose instead of
    emitting tool_calls at all, which is itself a useful data point when
    comparing backends, not a bug in this loop. Verify against your actual
    server/model before trusting the trace.

    Some models (e.g. phi4-mini-reasoning) inline <think>...</think> in
    message["content"] regardless of the "think" flag below, rather than
    using Ollama's separate "thinking" field the way deepseek-r1 does. By
    default (strip_thinking=True) this is stripped with the same
    _strip_thinking used by ollama_client's narration path, so a model
    that ignores the flag doesn't leak raw chain-of-thought into the
    reported final answer. Pass strip_thinking=False to see the raw
    content instead -- e.g. to check whether a model is thinking at all
    when it fails to call tools.

    think: forwarded to Ollama's "think" request field (default False,
    matching the prior hardcoded behavior). Set True to see whether
    letting a reasoning-capable model actually think before responding
    changes whether it calls tools at all, as opposed to just changing
    where its reasoning text ends up.
    """
    import requests

    tools = CircuitAnalystTools(algo)
    dispatch = _make_dispatch(tools)
    ollama_tools = _tool_schemas_to_ollama(TOOL_SCHEMAS)

    options = {}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []

    for _ in range(max_steps):
        payload = {
            "model": model,
            "messages": messages,
            "tools": ollama_tools,
            "stream": False,
            "think": think,
        }
        if options:
            payload["options"] = options
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        tool_calls = message.get("tool_calls") or []
        content = message.get("content", "") or ""
        if strip_thinking:
            truncated = data.get("done_reason") not in (None, "stop")
            content = _strip_thinking(content, truncated=truncated)

        messages.append({"role": "assistant", "content": content,
                          "tool_calls": tool_calls})

        if not tool_calls:
            return content, trace

        for call in tool_calls:
            fn_info = call.get("function", {})
            name = fn_info.get("name")
            args = fn_info.get("arguments") or {}
            fn = dispatch.get(name)
            if fn is None:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"error": str(e)}
            trace.append((name, args, result))
            if verbose:
                print(f"[tool call] {name}({args}) -> {json.dumps(result)[:200]}")
            messages.append({"role": "tool", "name": name, "content": json.dumps(result)})

    return "[max_steps reached without a final answer]", trace


def run_agent_aggregator(base_url, api_key, algo, question, model, max_steps=8, verbose=True,
                          timeout=180, max_tokens=1024, temperature=None):
    """OpenAI-compatible-API equivalent of run_agent()/run_agent_ollama(),
    for aggregators (e.g. "ModelProxy," a pseudonym -- see
    agentic_testing_notes.md) that speak real OpenAI tool-calling
    semantics over /chat/completions -- distinct from run_agent_ollama()
    because OpenAI's tool_calls carry a real "id" that the follow-up tool
    message must reference via "tool_call_id"; Ollama's format omits ids
    entirely and matches by "name" instead, so the two aren't
    interchangeable despite looking similar. See agentic_testing_notes.md
    for trust caveats about what a given aggregator model name actually
    routes to -- this function doesn't verify that.
    """
    import requests

    tools = CircuitAnalystTools(algo)
    dispatch = _make_dispatch(tools)
    openai_tools = _tool_schemas_to_ollama(TOOL_SCHEMAS)  # same shape OpenAI expects

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []

    for _ in range(max_steps):
        payload = {
            "model": model,
            "messages": messages,
            "tools": openai_tools,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        content = message.get("content") or ""

        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            return content, trace

        for call in tool_calls:
            fn_info = call.get("function", {})
            name = fn_info.get("name")
            try:
                args = json.loads(fn_info.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = dispatch.get(name)
            if fn is None:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"error": str(e)}
            trace.append((name, args, result))
            if verbose:
                print(f"[tool call] {name}({args}) -> {json.dumps(result)[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": json.dumps(result),
            })

    return "[max_steps reached without a final answer]", trace
