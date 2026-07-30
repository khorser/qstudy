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

# Testing compare_backends.py / ai_narrator.py against local Ollama models

Separate from the agentic tool-planning module above: this piece is
plain grounded narration -- the model is handed already-computed slice
facts and asked to explain them in 2-4 sentences, no tools, no planning.
Worth testing on its own since it's a different task shape and the
failure modes turned out to be different too.

```
pixi run -e qc python compare_backends.py --skip-anthropic --ollama-model <model>
```

## qwen3.5:9b and qwen2.5:7b: both grounded, neither hallucinated

Ran both against the `xor`-oracle Deutsch-Jozsa circuit, scored against
`DJ_BV_REFERENCE`. Both stayed within the given facts -- no invented gate
names or numbers, matching the design intent in `ai_narrator.py`'s
docstring. But the coverage scores are a weak proxy for actual quality in
both directions:

- **Coverage under-counts good narrations.** Both models scored 0.00 on
  the `init` slice (missing "ancilla," "eigenstate," "phase kickback,"
  "-1") despite giving factually accurate descriptions of what the raw
  facts show. Neither model was ever told *which* qubit is the ancilla or
  that phase kickback is happening -- `build_slice_prompt` doesn't label
  that -- so declining to use those words is arguably the *correct*
  response to avoid inventing a fact, not a narration failure. The
  reference answer only reads well because a human already knows the
  algorithm; the scorer conflates "matches expert vocabulary" with
  "grounded in what was actually given."
- **Coverage over-counts a real error.** On the `apply` slice,
  `qwen2.5:7b` first states the three CX gates "entangle some qubits,"
  then two sentences later says "there are no entangled pairs of
  qubits" -- a direct self-contradiction. `qwen3.5:9b`'s narration of the
  same slice is careful never to claim entanglement happened, only that
  "entangling gate counts show non-trivial interaction" while "no
  measurable entanglement entropy" was detected -- consistent with the
  actual physics (phase kickback via CNOTs onto an ancilla in `|->`
  leaves the state a product state, per the note already in
  `q/README.md`). Yet `qwen2.5:7b` scored *higher* (0.33 vs 0.00) on this
  slice, purely because it happened to use the word "oracle" elsewhere in
  the same paragraph. The keyword scorer doesn't notice the
  contradiction at all.

## phi4-mini-reasoning:3.8b: a real bug, not just a bad model

Ran the same comparison against `phi4-mini-reasoning:3.8b` (default
`max_tokens=300`, matching `explain_slice`'s default). Every single
slice's "narration" was a raw, mid-sentence `<think>...` block with no
closing tag -- e.g. for `init`: *"Density-matrix purity is 1, meaning all
qubits are perfectly known"* -- cut off mid-thought.

Root cause: `ollama_client.py`'s `_strip_thinking` regex,
`<think>.*?</think>`, only matches a **closed** block. This model needs
far more than 300 tokens to think through even one slice's facts, so
generation is cut off by `max_tokens` before a closing tag is ever
emitted. The regex found nothing to strip, so raw, truncated internal
monologue was returned as the "narration" -- and the coverage scorer then
scored that monologue (which incidentally mentions "hadamard" or
"oracle" while reasoning aloud about the general algorithm), producing
misleadingly non-zero coverage numbers for a narration that never
actually happened.

### Fix 1: `_strip_thinking` needs to know whether generation was truncated

Naively "discard everything after an unterminated `<think>`" turned out
to be wrong too. Diagnostic direct call to `/api/chat` with
`num_predict: 3500` showed this model can finish naturally
(`done_reason: "stop"`, ~1838 tokens) -- but **still never emits a
closing `</think>` tag**, even when done. Its reasoning and its answer
aren't separated by any reliable delimiter; the whole response is one
continuous block. Discarding "from `<think>` to end of string"
unconditionally would throw away the real answer along with the
reasoning whenever this model is used, not just when it's cut off.

Fixed by giving `_strip_thinking` a `truncated` parameter (both
`ollama_client.py` and `agentic_analyst.py`'s `run_agent_ollama` now pass
`data.get("done_reason") not in (None, "stop")`):
- **truncated=True** (budget ran out mid-thought, `done_reason != "stop"`):
  discard from the unterminated `<think>` through end of string -- there's
  no real answer in what's left.
- **truncated=False** (model stopped on its own but still left the tag
  open): only strip the bare `<think>` token itself and keep the rest --
  better to show a slightly messy blended reasoning+answer text than to
  silently discard a real answer that happens to follow an unclosed tag.

### Fix 2: `--max-tokens` CLI flag on `compare_backends.py`

Added `--max-tokens` (default 300, unchanged), threaded through
`run_backend` -> `explain_slice`, rather than hardcoding a bump for one
model name in shared code. Testing progression:
- `max_tokens=300` (default): "Empty response" for every slice --
  correctly identified as truncated, stripped to nothing, and the
  existing fallback message in `ollama_client.py` fired as designed
  ("...max_tokens ran out during reasoning...").
- `max_tokens=1500`: still "Empty response" for every slice -- 1500
  tokens is *still* not enough for this model to finish even one slice.
- `max_tokens=3500`: **`prep` and `apply` slices now produce real,
  substantive narrations**, correctly extracted (verified no leaked
  `<think>` tag, no truncated fragment). `init`/`done`/`final` instead
  hit `OllamaClient`'s separate, unrelated 120s request timeout -- this
  model consistently takes 100-200+s per slice once given enough budget
  to actually finish, and `OllamaClient.__init__`'s `timeout=120` default
  (used by `compare_backends.py`'s `build_ollama_client`, separate from
  `run_agent_ollama`'s own `timeout` param) isn't long enough. Not yet
  addressed -- a straightforward `--ollama-timeout` CLI flag would fix it
  the same way `--max-tokens` did, but wasn't part of what was asked for
  this round.

### Net result

`phi4-mini-reasoning:3.8b` isn't simply "bad at this task" the way it was
at tool planning -- it's a *much* higher token-budget model than the
others tested (qwen3.5:9b and qwen2.5:7b both answer in a few hundred
tokens; this one needs 1500-3500+ per slice), and the original code had
two latent bugs that made that show up as silent hallucination-shaped
garbage instead of an obvious "budget too small" error: an unstrippable
unterminated `<think>` tag, and a scorer that can't tell the difference
between a real narration and raw internal monologue that happens to share
vocabulary with one. Both are now fixed at the harness level.

## Follow-up: bumped the Ollama client timeout, re-ran at max_tokens=3500

`OllamaClient.__init__`'s default `timeout=120` was too short once
`--max-tokens` was raised enough for this model to actually finish (it
routinely takes 100-200+s per slice). Bumped the default to 300s, and
added a matching `--ollama-timeout` CLI flag to `compare_backends.py`
(mirroring `--max-tokens` rather than hardcoding a per-model value).

Re-ran `phi4-mini-reasoning:3.8b` at `--max-tokens 3500 --ollama-timeout
300`. Result: 3 of 5 slices (`init`, `prep`, `done`) now produce genuine,
complete narrations instead of raw unterminated reasoning or a timeout
exception; `apply` and `final` still ran out of the 3500-token budget
before finishing (correctly reported as "Empty response," not leaked
partial text) -- confirming this model's per-slice token need varies
enough that there's no single reliable budget, only "enough, most of the
time." No more hard `ReadTimeout` crashes anywhere in this run.

Two further things surfaced by actually reading the completed narrations:

- **A real factual error in a "successful" narration.** For `init`, the
  facts given are `gate_counts: {"x": 1}` -- one Pauli-X gate, applied to
  the ancilla per the actual circuit (`c.x(y)` in `DeutschJozsa.get_circuit`).
  The model instead reasons "x=1... I think refers to applying the
  Hadamard gate" and concludes the slice "applies Hadamard gates to each
  input qubit" -- fabricating a completely different (and wrong) gate and
  target qubit from a plain, unambiguous key in the JSON it was given.
  Same species of error as `qwen2.5:7b`'s entangling-gate mixup in the
  agentic test above: correct data in, confidently wrong claim out, no
  tool-use or planning involved this time -- it's a base
  reading-comprehension failure on structured facts, independent of the
  tool-calling story entirely.
- **A residual stripping bug**: the `done` slice's raw response contained
  a second, spurious bare `<think>` reopening mid-answer (visible as a
  garbled `</> <think>` before a "**Step-by-Step Explanation:**" section),
  which leaked through because the fix only stripped the *first* bare
  `<think>` occurrence (`text.replace("<think>", "", 1)`). Changed to
  strip all bare occurrences (`text.replace("<think>", "")`) since this
  model has now been observed doing this more than once in a single
  response. Verified against a synthetic reproduction of the exact
  pattern seen in that response.

Net effect of this whole `phi4-mini-reasoning:3.8b` thread: what looked
at first like "this model just can't do the task" turned out to be mostly
a token-budget and harness-bug problem (now fixed), *plus* a genuine,
separate reading-comprehension weakness that shows up even once budget
and stripping are no longer confounding the picture. Worth keeping both
framings in the writeup -- they're different claims and this model
exhibits both.

## Run 5: llama3.1:8b -- inconsistent, not a clean pass or fail

Every clean tool-planning result up to this point was Qwen-family
(`qwen3.5:9b`, `qwen2.5:7b`); every failure was a reasoning model
(`phi4-mini-reasoning`, `deepseek-r1`) -- confounded, since it was unclear
whether "reliably plans tool calls" tracks the Qwen family specifically or
"non-reasoning, tool-call-trained model" more generally. `llama3.1:8b`
ships with Meta's native function-calling template, so it's a same-lineage
sanity check on whether the Qwen result generalizes.

Ran the identical question twice (same code path, `max_steps=10`,
`timeout=180`):

- **Run A (11.3s)**: called `compare_resource_cost('0', 'xor')`
  *immediately*, skipping `run_circuit` -- correctly caught by the
  guardrail (`"error": "oracle '0' not yet simulated -- call run_circuit
  first"`). The model's next message correctly diagnosed the problem in
  prose ("It seems I need to simulate the circuit... Let me do that...")
  but then wrote **`run_circuit()` as literal text** instead of emitting
  an actual tool call. Since that response contained no `tool_calls`, the
  loop terminated immediately, treating the placeholder text as the final
  answer. A distinct failure mode from anything seen so far: not zero
  tool calls, not a hallucinated circuit -- one *wrong-order* tool call,
  a correct verbal diagnosis of its own mistake, and then a failure to
  actually act on that diagnosis.
- **Run B (12.6s)**: clean -- `run_circuit` x2, then `compare_resource_cost`
  directly, correct grounded final answer. Textbook ideal trace, as good
  as `qwen3.5:9b`/`qwen2.5:7b`'s best runs.

Conclusion: `llama3.1:8b` *can* plan this task correctly and does ship
with real tool-calling support, but is run-to-run non-deterministic
between a clean plan and a broken one that doesn't recover from its own
ordering mistake. That's a third distinct category, next to "consistently
plans well" (Qwen models tested) and "never engages tools at all"
(phi4-mini-reasoning) -- worth naming explicitly in the writeup rather
than averaging it into either bucket.

## Run 6: gemma2:9b -- rejected before it ever reaches the model

Immediate `400 Bad Request` from Ollama itself:

```
{"error": "registry.ollama.ai/library/gemma2:9b does not support tools"}
```

This is qualitatively different from every other failure recorded here.
`phi4-mini-reasoning:3.8b` and `llama3.1:8b`'s bad run both got a normal
`200` response and a real (if wrong) attempt from the model -- the
request reached the model and the model's own behavior was the problem.
`gemma2:9b` never gets that far: Ollama's chat template for this model
has no tool-calling scaffolding at all, so the API refuses the request
outright, before the model sees the question or the tools.

This is the cleanest confirmation of the hypothesis that motivated
testing these two models: the axis that actually predicts success here
isn't "reasoning vs. non-reasoning" -- it's whether the model was trained
with, and Ollama has wired up, native tool-calling support at all. Gemma 2
wasn't; Llama 3.1 and the Qwen2.5+/3.5 line were.

## Updated cross-model summary

| Model | Tool-calling available? | Result |
|---|---|---|
| qwen3.5:9b | yes | Consistently clean, grounded plans across multiple runs |
| qwen2.5:7b | yes | Clean plan, but one run contained a confidently-wrong misread of correct tool data |
| llama3.1:8b | yes | Non-deterministic: one clean run, one run with a wrong-order call it correctly diagnosed but failed to correct |
| phi4-mini-reasoning:3.8b | yes (template accepts it) | Model itself never calls tools; separately, needs a much larger token budget than other models even for plain narration, and can misread structured facts outright |
| deepseek-r1:8b | yes (native thinking field) | Untested for planning quality -- abandoned as impractically slow per turn on this hardware |
| gemma2:9b | **no** -- Ollama rejects the request | N/A -- never reaches the model |

The overall shape of the result: whether a small local model can serve as
the planner in this kind of narrow, guarded agentic loop depends far more
on whether it was trained with (and Ollama has wired up) tool-calling
support than on model size or "reasoning" branding -- and even among
models that clear that bar, reliability varies from "consistent" to
"coin flip" to "confidently wrong," which is exactly the kind of variance
a one-shot demo would hide and a portfolio writeup should surface
explicitly rather than reporting a single cherry-picked run.

## Non-agentic cross-check: llama3.1:8b and gemma2:9b on plain narration

The agentic runs above only tested `llama3.1:8b`/`gemma2:9b` on
tool-planning. `gemma2:9b` failed that outright (Ollama rejects `tools`
for it entirely) -- but that says nothing about whether the model itself
is competent at the *other* task in this portfolio piece, plain grounded
narration (`compare_backends.py`/`ai_narrator.py`), which never sends
Ollama a `tools` field at all. Ran both through the same
Deutsch-Jozsa-`xor` narration comparison used earlier for qwen3.5:9b /
qwen2.5:7b / phi4-mini-reasoning.

- **`llama3.1:8b`** narrates competently overall (similar coverage
  profile to the Qwen models), but repeats the *exact same* misread seen
  earlier with `phi4-mini-reasoning:3.8b`: for `init`, it states *"a
  single Hadamard gate (x=1) is applied to one qubit"* -- but `x=1` means
  one Pauli-X gate (the ancilla flip in `c.x(y)`), not a Hadamard.
  (Revised after a third data point below: `qwen2.5:7b` and
  `deepseek-r1:8b` both correctly identify this as Pauli-X, and those are
  also the two strongest, most consistent performers across every test
  in this file. So this looks less like a prompt-format trap that
  catches everyone equally and more like it correlates with general
  model competence -- weaker models guess "Hadamard" because
  Deutsch-Jozsa is strongly associated with Hadamards, stronger models
  actually parse the literal key. `build_slice_prompt`'s terse `"x=1"`
  notation is still plausibly a contributing factor -- a clearer label
  like `"X gate: 1"` would remove the ambiguity entirely and cost
  nothing -- but it's not the sole explanation.)
- **`gemma2:9b`** narrates solidly, and notably *does not* make the
  x=1/Hadamard mistake -- it correctly describes `init` as "a single
  qubit initialized to the state |1>," and hedges appropriately
  elsewhere ("likely," "would be needed to know precisely") rather than
  asserting unsupported specifics. No factual errors spotted in any of
  the five slices.

This is the clean confirmation that tool-calling support and narration
competence are separate axes, not the same underlying capability:
`gemma2:9b` fails the agentic task completely (hard API rejection, never
reaches the model) but is a perfectly reasonable narrator when just asked
to explain facts it's handed directly. Model choice for this portfolio
piece should be task-specific -- the same model ranking does not carry
over between "plan which tools to call" and "narrate given facts."

## deepseek-r1:8b on narration: task shape changes whether "too slow" applies

`deepseek-r1:8b` was abandoned for the agentic loop as impractically
slow (didn't respond even at a 1200s timeout to a single non-tool
prompt). But narration is a single request per slice, not up to 10
chained tool-call round trips -- worth checking separately rather than
assuming the same verdict carries over. Ran with a generous budget
(`--max-tokens 2000 --ollama-timeout 900`).

First pass (`--max-tokens 2000`): **4 of 5 slices completed** with real,
reasonably grounded narrations (`init`, `prep`, `apply`, `final`). Only
`done` ran out of the 2000-token budget mid-thought -- and it failed
*safely*: because this model uses Ollama's native, separate `"thinking"`
field (unlike `phi4-mini-reasoning`, which inlines `<think>` in `content`
and never closes it), `ollama_client.py`'s existing fallback message fired
exactly as designed ("...only a 'thinking' field... try a much higher
max_tokens..."), with no leaked or truncated garbage presented as an
answer.

Re-ran at `--max-tokens 3500` (the same budget that got
`phi4-mini-reasoning` mostly working): **all 5 slices completed**,
including `done` this time. Quality held up across the full set --
`init` again correctly identifies "an X (NOT) gate... flipping its value
from |0> to |1>" (no Hadamard misread, consistent with the first pass),
and every slice stays cautious, declining to overclaim entanglement or
invent oracle mechanics beyond what the facts support. `done`'s coverage
score is 0.00 despite being a real, sensible answer ("applies 3 gates to
prepare for measurement... no entangling gates... no cross-talk") --
another instance of the scorer penalizing a grounded but
vocabulary-sparse narration, same pattern noted for `qwen3.5:9b`/
`qwen2.5:7b` above.

Net: "too slow to use" is a claim about a specific task shape (a long,
guarded multi-turn agentic loop), not a blanket verdict on the model. For
a single-shot grounded-narration task, `deepseek-r1:8b` is slow but fully
usable at a large-enough token budget (3500, same as
`phi4-mini-reasoning` needed) and produces some of the more careful,
differentiated narrations recorded in this file -- worth remembering
before writing off a model based on one task's results.
