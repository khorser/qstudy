# Testing agentic_analyst.py against local Ollama models

Notes from running the tool-use planning agent (`agentic_analyst.py` /
`compare_agent_backends.py`) against models already pulled locally via
Ollama, instead of (or alongside) Claude.

Environment: qiskit and `requests` aren't in the base interpreter for this
repo -- use the `qc` pixi environment from `~/Src/qq`:

```
cd ~/Src/qq/qstudy
pixi run -e qc python compare_agent_backends.py --skip-anthropic --ollama-model <model> --max-steps 10
```

## Run 1: qwen3.5:9b

Question: *"Compare the resource cost of the '0' and 'xor' oracles for
this Deutsch-Jozsa instance, and tell me specifically where the
difference comes from."*

Trace (8 tool calls, 0 errors):

```
run_circuit(oracle='0')
run_circuit(oracle='xor')
get_slice_facts(oracle='0', label='init')
get_slice_facts(oracle='xor', label='init')
get_slice_facts(oracle='0', label='prep')
get_slice_facts(oracle='xor', label='prep')
get_slice_facts(oracle='0', label='apply')
get_slice_facts(oracle='xor', label='apply')
```

Final answer:

> The difference in resource cost between the '0' and 'xor' oracles comes
> entirely from their **apply** slices:
> - The **'0'** oracle has an empty apply slice (no gates, depth 0), as it
>   represents a constant function returning all zeros.
> - The **'xor'** oracle performs three CNOT entangling gates in its apply
>   slice to implement the parity-based f(x) = x_1 ⊕ ... ⊕ x_n mapping,
>   resulting in additional gate counts and increased circuit depth.

Correct and properly grounded in the tool results -- matches what
`resource_stats.py` actually reports for these two oracles.

### Was this genuinely self-planned, or scripted?

Nothing in the code prescribes this sequence:

- `AGENT_SYSTEM_PROMPT` only describes what the tools do and that
  `run_circuit` must precede inspection for a given oracle -- it never
  says "call list_oracles first," "go slice by slice," or "use
  compare_resource_cost." The only enforcement is a runtime guard:
  `get_slice_facts` / `compare_resource_cost` error out if `run_circuit`
  hasn't been called yet for that oracle (`agentic_analyst.py:102-106`).
- **It skipped `list_oracles` entirely.** The question already named `'0'`
  and `'xor'` in prose; the model parsed those out and went straight to
  `run_circuit` rather than discovering the oracle list first.
- **It chose an interleaved, breadth-first strategy**: simulate both
  oracles, then walk `init` -> `prep` -> `apply`, alternating oracle by
  oracle at each slice rather than fully draining one oracle before the
  other. Nothing forces that interleaving.
- **It stopped itself early.** It never inspected `done`/`final` --
  decided at `apply` that it had enough evidence and ended the loop on
  its own (no `max_steps` cutoff hit, no code forcing early stop).
- **It reinvented the comparison instead of using the built-in tool.**
  `compare_resource_cost(oracle_a, oracle_b)` exists for exactly this
  question; the model instead diffed the two oracles' facts itself via
  repeated `get_slice_facts` calls. `used_compare_tool: False` in the
  eval output. The final answer was still correct, but this is a less
  efficient plan than the "intended" one -- worth calling out explicitly
  in any writeup rather than only reporting the eval table, since
  `used_compare_tool: False` reads as a negative result unless you frame
  it as "reasoned around the tool," not "failed to use the tool."

### Caveat found while picking the next model

`run_agent_ollama` hardcodes `"think": False` in every request
(`agentic_analyst.py:311`), with no parameter to override it. That mirrors
the deliberate choice in `ollama_client.py` for `ai_narrator.py` (short
grounded narration doesn't benefit from hidden chain-of-thought). But
`agentic_analyst.py`'s own module docstring calls this loop out as *"the
one place in this portfolio piece where a reasoning model is worth
benchmarking against a standard one"* -- and a reasoning model tested with
its thinking mode forced off isn't really being tested on the thing that
distinguishes it. This is a real inconsistency between the stated purpose
of the module and what the Ollama backend actually does; worth deciding
deliberately (add a `think` toggle, or note explicitly that this
benchmark evaluates *planning without visible CoT* by design) rather than
leaving it as an unexamined default.

## Run 2: deepseek-r1:8b -- abandoned, impractically slow on this hardware

Attempted the same question against `deepseek-r1:8b`. First full-loop attempt
timed out at 600s (the CLI script's hardcoded per-request timeout in
`run_agent_ollama`). Raised the timeout progressively (300s -> 600s ->
1200s) calling `requests.post` directly against `/api/chat` -- still no
response. To isolate whether the tool schemas were the problem, sent the
*same system prompt + question with no `tools` field at all*: still no
response within 100s, versus 13s for a trivial "Say OK" prompt on the same
model/machine (GPU-resident per `ollama ps`).

Conclusion: this isn't a timeout-tuning problem, it's that `deepseek-r1:8b`
takes multiple minutes minimum per turn on any non-trivial prompt on this
machine, independent of tool-calling. A full agent run (up to 10 turns)
would plausibly take 20-60+ minutes. Not a useful data point for this
exercise -- abandoned in favor of a smaller reasoning model.

## Run 3: phi4-mini-reasoning:3.8b -- clean failure, not a close call

Same question, `max_steps=10`, `timeout=300`. Completed in 206s, but the
result is a clear failure rather than a weaker version of qwen3.5:9b's
answer:

- **Trace is empty.** Zero tool calls -- no `list_oracles`, no
  `run_circuit`, nothing. The model answered directly on the first turn
  despite `AGENT_SYSTEM_PROMPT` explicitly saying "ground every claim in
  tool results" and "you do NOT have access to the raw circuit."
- **It fabricated a wrong circuit model from scratch.** The "final answer"
  invents two registers `A`/`B`, Toffoli gates, T-gate counts -- none of
  which describe the actual `DeutschJozsa` circuit in this file (one
  ancilla qubit, plain `cx` gates, no Toffolis, no T-gates at all). This
  is exactly the ungrounded-hallucination failure mode the tool-based
  design is meant to prevent, and it happened with zero tool calls to even
  ground against.
- **`"think": False` is not honored by this model.** `deepseek-r1:8b`
  routes its reasoning into a separate `thinking` JSON field when asked;
  `phi4-mini-reasoning:3.8b` inlines `<think>...</think>` straight into
  `content` regardless of the flag. `run_agent_ollama` never strips this
  (only `ollama_client.py`'s `_strip_thinking`, used by the narrator path,
  does that) -- so ~200 lines of raw, rambling chain-of-thought are
  reported as the "final answer" verbatim.
- **It degenerates into incoherent fragments by the end** ("So. But. (1).
  And in the. Now. If...") rather than converging on a conclusion --
  reads like it ran out of context budget mid-ramble and the loop
  accepted whatever came back as a valid final answer, because with no
  tool calls in the response there was nothing to distinguish "still
  planning" from "done."

### Net comparison across the three models tried

| Model | Tool calls | Errors | Final answer | Verdict |
|---|---|---|---|---|
| qwen3.5:9b | 8 (3 after prompt strengthening, see below) | 0 | Correct, grounded, concise | Real planning, initially skipped `compare_resource_cost` but got there anyway; used it directly after the prompt change |
| qwen2.5:7b | 3 | 0 | Mostly correct but contains one factual error (see Run 4) | Right tool, right data retrieved, but misread the JSON and asserted something the data doesn't say |
| deepseek-r1:8b | -- | -- | -- | Unusable on this hardware -- too slow per turn to finish |
| phi4-mini-reasoning:3.8b | 0 | -- | Fabricated, incoherent, includes leaked `<think>` text | Never engaged the tools at all |

This is a stronger portfolio point than "qwen3.5:9b did fine": the same
tool-calling harness, same guardrails, same system prompt produced a
well-grounded answer from one small local model and a fully hallucinated,
tool-free answer from another. That gap -- not the existence of tools --
is the evidence that the agentic design is doing real work here, and it's
also evidence that "reasoning model" is not a reliable proxy for "will
actually plan tool calls" at this parameter scale.

### Follow-up worth doing before treating this as final

- [x] Strip `<think>` blocks from `run_agent_ollama`'s reported final
  answer and trace, matching `ollama_client.py`'s narrator path, so a
  model that ignores `think: False` doesn't pollute the displayed output.
  Done: `run_agent_ollama` now imports `_strip_thinking` from
  `ollama_client.py` and applies it to `message["content"]` before
  appending to history or returning, gated behind a new
  `strip_thinking=True` parameter (pass `False` to see raw content, e.g.
  to check whether a model is thinking at all when it skips tool calls).
  Re-ran `phi4-mini-reasoning:3.8b` against the same question afterward:
  final answer is now clean prose with no leaked `<think>` text (102s
  this time, vs 206s before -- some run-to-run variance, unrelated to the
  fix). Trace is still empty -- it still called zero tools -- but this
  time it fabricated a *different* kind of nonsense: invented terminology
  ("H-object," "W-object," "hybrid adversary") not found anywhere in this
  codebase or in Deutsch-Jozsa theory, landing on the qualitatively right
  answer (constant cheaper than balanced) for entirely made-up reasons.
  Confirms the stripping fix is working as intended and is orthogonal to
  the tool-call failure -- that failure is a genuine planning gap, not an
  artifact of `<think>` leakage.
- [x] Re-test `phi4-mini-reasoning:3.8b` with `think: True` explicitly to
  see whether the tool-call failure is a "thinking mode ignored" artifact
  or a genuine planning failure independent of that setting. Done: added
  a `think` parameter to `run_agent_ollama` (default `False`, matching the
  prior hardcoded value) and called it with `think=True`. Ollama rejects
  the request outright:
  `{"error": "\"phi4-mini-reasoning:3.8b\" does not support thinking"}`
  -- this model doesn't declare native thinking support in its Ollama
  template at all, so the `<think>` text it emits is baked into every
  response regardless of any API toggle; there is no "give it its
  thinking back" experiment available for this model. Confirms the
  zero-tool-calls result is a genuine planning failure, not an artifact
  of suppressed reasoning. (For contrast, `deepseek-r1:8b` *does* support
  the toggle -- it returned a separate `"thinking"` field in an earlier
  ad hoc test -- but that model was abandoned above for being
  impractically slow on this hardware.)
- [x] Consider a stricter system prompt or a forced first tool call for
  small models that skip tool use entirely. Done: strengthened
  `AGENT_SYSTEM_PROMPT` (shared by both `run_agent` and
  `run_agent_ollama`) with an explicit, blunt requirement --

  > You must call at least one tool before writing a final answer -- you
  > have no other source of information about this specific circuit, and
  > anything you recall about Deutsch-Jozsa in general will not match the
  > actual oracle definitions or gate structure used here. If you are
  > about to answer without having called any tool yet in this
  > conversation, call a tool instead.

  Re-ran both models against the same question:

  - **`phi4-mini-reasoning:3.8b`: no change.** Still zero tool calls
    (156s). It ignored the instruction entirely and fabricated a *third*
    wrong circuit model -- this time a Bernstein-Vazirani-flavored
    hidden-string parity check with `O(n^2)` CCNOTs and extra ancillae,
    none of which exist in this circuit (plain CX gates from the input
    register to one ancilla, no hidden string, no CCNOTs at all). Three
    different fabrications across three runs (Toffoli/T-gate soup,
    "H-object"/"W-object" nonsense, now BV-style parity) is itself a
    signal: this isn't converging toward the right answer by chance, it's
    unconstrained pattern-completion over "things that sound like
    Deutsch-Jozsa oracle circuits," with nothing tying it to what
    `run_circuit`/`get_slice_facts` would actually report if called.
  - **`qwen3.5:9b`: no regression, and one improvement.** Still correct
    and fully grounded, but this time it went straight for the intended
    `compare_resource_cost` tool (3 tool calls: `run_circuit` x2 then
    `compare_resource_cost`, vs. 8 calls and a hand-rolled comparison
    before the prompt change). Small sample, so this could be run-to-run
    variance rather than the prompt directly causing more efficient
    planning -- but at minimum the stricter instruction didn't hurt a
    model that was already behaving well.

  Conclusion, matching the note this item started from: a stricter
  system prompt is a free lever worth pulling (it didn't cost
  `qwen3.5:9b` anything and arguably helped), but it is not a fix for a
  model that doesn't reliably plan tool calls at all --
  `phi4-mini-reasoning:3.8b` will apparently answer confidently and
  ungrounded no matter how explicitly it's told not to. Treating this as
  a deliberate scope boundary rather than a bug: this portfolio piece's
  tool surface is intentionally narrow and safe (no code execution,
  guarded call ordering), and the interesting result is precisely that
  the harness can't paper over a model that won't engage the tools --
  which model you point an agentic loop at matters as much as the loop
  itself.

## Run 4: qwen2.5:7b -- the non-reasoning baseline, for contrast with qwen3.5:9b

`compare_agent_backends.py` already defaults to this model; worth actually
running it once there was a real qwen3.5:9b result to compare against, to
separate "is this a reasoning-vs-non-reasoning story" from "is this
specific to the qwen3.5 generation." Same question, strengthened system
prompt in place, `max_steps=10`.

Result: 3 tool calls, 21s total, 0 tool errors --

```
run_circuit(oracle='0')
run_circuit(oracle='xor')
compare_resource_cost(oracle_a='0', oracle_b='xor')
```

Fast, efficient, went straight for the intended comparison tool -- as good
a plan as `qwen3.5:9b` produced after the prompt strengthening, on a
smaller/older model. That's a useful data point on its own: this task
apparently doesn't need "reasoning" or even the newest model generation,
it needs a small-but-competent instruction-following model with decent
native tool-calling training. `phi4-mini-reasoning:3.8b`'s total failure
looks like a model/training gap specific to that model, not a "small
local models can't do this" story.

But the final answer isn't fully correct, and the error is a different
species from anything seen so far. It states: *"the 'xor' oracle has 3
entangling gates in both the 'apply' and 'done' slices."* Checking the
actual `compare_resource_cost` result it was given:

```json
"done": {
  "entangling_gates": {"0": 0, "xor": 0},
  "total_gates_this_slice": {"0": 3, "xor": 3}
}
```

Both oracles have **zero** entangling gates in `done` -- it's just three
non-entangling Hadamards (the final `h(x)` round before measurement). The
model appears to have conflated `total_gates_this_slice: 3` with
`entangling_gates` when reading the JSON, and asserted something the tool
result doesn't say. Unlike `phi4-mini-reasoning:3.8b`'s wholesale
invention of a circuit it never inspected, this is a *misread of real,
correctly-retrieved data* -- the tool call happened, the right facts came
back, and the answer is still subtly wrong in a way that reads as
confident and well-formatted. That's arguably the more dangerous failure
mode for a resource-estimation tool to have in practice: a fabrication
from a model that skipped tools entirely is easy to catch by noticing the
trace is empty; a plausible near-miss from a model that *did* call the
right tool is the kind of error that survives a quick review.

### Updated conclusion

Across four models, the interesting axis isn't reasoning vs. non-reasoning
or model size -- it's whether the model reliably (a) calls tools at all,
and (b) reads their JSON output correctly once it has it. `qwen3.5:9b` and
`qwen2.5:7b` both clear (a); only `qwen3.5:9b` was verified clean on (b)
across two runs. `qwen2.5:7b` clears (a) but failed (b) on this run. Worth
noting for the writeup: "the model used the tools" is necessary but not
sufficient for a grounded answer -- the coverage-scorer idea from
`ai_narrator.py`'s README (score a narration against hand-verified facts)
would catch exactly this class of error and is the natural next thing to
port over to this module if it goes further than a single-question demo.
