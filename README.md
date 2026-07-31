# qstudy
My quantum computing studies grown out into a generic CircuitSlicer and then
into an AI-Assisted Slicer with Resource Estimation and Grounded Narration.

You will need Jupytext to convert notebooks back to .ipynb using a command like `jupytext --to notebook fundamentals-of-quantum-algorithms.py`

https://github.com/user-attachments/assets/d5b502dc-6c89-4bd9-929e-06d4391da9dd

## Where this started

`CircuitSlicer` (in `qstudy.py`) wasn't built as a product — it grew out of
wanting to actually understand what's happening inside quantum algorithms,
step by step, rather than just running a circuit and reading off the final
counts. That shows in the design: barrier-delimited slicing means any
algorithm can be inspected the same way without writing custom
instrumentation per algorithm, and it's been reused as-is across Deutsch,
Deutsch-Jozsa, Bernstein-Vazirani, Simon (with GF(2) postprocessing),
Grover, phase estimation, teleportation, superdense coding, CHSH, and a set
of custom multi-controlled/fanout gate constructions — one widget class,
a dozen very different algorithms.

A few things in there are worth calling out on their own merits, independent
of the AI layer below:
- `factor_out()` — extracting a common symbolic factor (including
  irrational ones like `sqrt(2)`, `sqrt(3)`) across a whole matrix via
  `sympy.nsimplify` + rational-gcd bookkeeping is a genuinely fiddly bit of
  algebra to get right, and it's what makes the printed statevectors/matrices
  actually readable instead of a wall of floating-point noise.
- Per-slice `Operator` reconstruction from the *incremental* circuit
  between barriers, rather than re-deriving it from the full circuit each
  time, is a clean way to get "what did just this step do" without
  re-simulating from scratch.
- The pairwise + full entanglement-entropy readout per slice turns "is
  this entangled" from a yes/no into an actual quantitative per-qubit-pair
  picture, which is more than most teaching tools bother with.

One caveat worth stating plainly, for anyone reading this as a portfolio
piece rather than just running it: the statevector and density matrix
shown for each slice come from `AerSimulator`'s `save_statevector` /
`save_density_matrix` instructions — a simulator-only introspection hook.
On real hardware there's no equivalent; you can't peek at the state
mid-circuit without collapsing it, you only get one classical bit-string
per shot at the end. So everything CircuitSlicer displays is "what the
simulator computed the state to be," not something you could ever directly
observe on a QPU. That's not a limitation of this tool specifically — it's
true of any statevector simulator — but it's the kind of distinction worth
being explicit about, and it's part of why gate/depth-based resource
counting (below) matters in the first place: it's a proxy you *can*
compute without needing to observe anything.

## AI-Assisted CircuitSlicer: Resource Estimation + Grounded Narration

An additive extension to `qstudy.py`'s `CircuitSlicer` — an interactive,
barrier-delimited circuit-stepping tool already used across your
Deutsch-Jozsa, Bernstein-Vazirani, Simon, Grover, and teleportation demos.
`qstudy.py` is untouched; everything here is a subclass + standalone
analysis modules layered on top.

### What's new (and who wrote it)

Everything below — `resource_stats.py`, `ai_narrator.py`,
`agentic_analyst.py`, `ai_circuit_slicer.py`, `aggregator_client.py`,
`ollama_client.py`, `headless_facts.py`, `compare_backends.py`,
`compare_agent_backends.py` — was written with Claude, in a working
session where each piece was actually run and checked rather than written
blind (see "Verified, not just written" below).

**`resource_stats.py`** — a static resource-estimation pass over the same
barrier-delimited slices CircuitSlicer already uses. For each slice it
reports gate counts by type, circuit depth, and entangling (>=2-qubit) gate
count, both per-slice and cumulative. Custom/opaque gates (e.g. an oracle
appended via `.to_gate()`) are unrolled to primitives before counting — a
first pass without this reported `circuit-42: 1` for the whole oracle,
which is exactly the kind of hidden-cost problem a resource estimator has
to catch. Standard gates (h, x, cx, ...) are deliberately left alone so
counts stay readable.

**`ai_narrator.py`** — builds a prompt from *only* facts already computed
by CircuitSlicer / resource_stats (gate counts, entanglement entropy,
purity, depth) and asks an LLM to narrate what a slice does and why. The
LLM never sees the raw circuit and is explicitly told not to invent
numbers beyond what's given — grounding by construction, not by
instruction alone. Includes a small hand-written reference (expert
baseline) for the Deutsch-Jozsa/Bernstein-Vazirani slices and a coverage
scorer, tested to confirm it scores the reference itself at 1.0 and a
deliberately vague narration at 0.0.

**`agentic_analyst.py`** — the open-ended counterpart to `ai_narrator.py`.
Instead of a fixed prompt over one slice's facts, the model gets a small
set of tools (`list_oracles` / `run_circuit` / `get_slice_facts` /
`compare_resource_cost`) and an open-ended question, and has to plan its
own multi-step investigation — genuine tool-call planning, not a template,
though still narrow: no arbitrary code execution, just those four
known-safe primitives. `run_agent` / `run_agent_ollama` /
`run_agent_aggregator` cover Anthropic's, Ollama's, and OpenAI-style
tool-calling shapes respectively (the latter two aren't interchangeable —
OpenAI-style `tool_calls` carry an id that the follow-up message must
reference; Ollama's doesn't).

**`ai_circuit_slicer.py`** — `AICircuitSlicer(CircuitSlicer)`: adds a
"Resources" tab and an "AI Explain this slice" button, plus an in-widget
backend selector across three backends — Anthropic, a local Ollama
server, or an OpenAI-compatible aggregator — each with its own "Refresh
models" button that queries the real, current model list live (Anthropic's
`models.list()`, Ollama's installed tags, or the aggregator's `/models`)
instead of guessing model names. Degrades gracefully with no API
key/server configured.

**`aggregator_client.py`** — duck-types the same client interface as
`ollama_client.py`, but speaks an OpenAI-compatible `/chat/completions` +
`/models` API instead, for testing narration against a third-party model
aggregator. Deliberately takes `base_url`/`api_key` as parameters (or
`OPENAI_BASE_URL`/`OPENAI_API_KEY` env vars, the standard names the
official `openai` SDK reads) rather than hardcoding a hostname.

**`ollama_client.py`** — duck-types just enough of the Anthropic client
interface to run `ai_narrator` against a local Ollama model instead.
Explicitly disables native "thinking" mode (`think: false`) for
reasoning-capable models (Qwen3, DeepSeek-R1, ...): this is a short
grounded-narration task, not one that benefits from chain-of-thought, and
leaving thinking on risks the whole token budget being spent on invisible
reasoning before any visible answer comes back. Also strips `<think>`
blocks and falls back to a `thinking` field if `content` still comes back
empty, so a misconfigured reasoning model fails loud instead of silent.

**`headless_facts.py`** — the same per-slice facts (gate counts,
entanglement, purity) `AICircuitSlicer` computes, but with zero
`ipywidgets`/`IPython.display` dependency. `CircuitSlicer` calls
`display()` at the end of its constructor, which needs a real Jupyter
frontend — fine in a notebook, but it means the widget can't be
instantiated from a plain script. This module re-simulates the same
barrier-delimited slices standalone, so `compare_backends.py` (below) runs
under plain `python`, not just inside a notebook.

**`compare_backends.py`** — runs the same slices through Claude and a
local Ollama model side by side, scores both against the hand-written
reference, and prints a coverage table plus narration-by-narration
comparison. Either backend is skipped gracefully if unreachable.

**`compare_agent_backends.py`** — the `agentic_analyst.py` equivalent:
runs the same open-ended question through Claude and a local Ollama model
and compares the *traces* (step count, which tools got used, whether
anything errored), since unlike narration there's no single "correct"
answer to score against.

`agentic_testing_notes.md` logs a much broader run of both scripts against
six local Ollama models and, via the aggregator above, several frontier
models (GPT-5.4, Gemini 3.1/3.5). The standout finding: most of the
aggregator's `claude-*`-labeled endpoints turned out to self-identify as
"Kiro" (an AWS product) rather than Claude when asked directly — the notes
document the identity-probe technique that caught this, worth reusing
before trusting any aggregator's named-model results again.

### Verified, not just written

Ran end-to-end against a reconstructed `DeutschJozsa` class:
- oracle-gate resolution: `f_xor` oracle correctly resolves from an opaque
  `circuit-N` blob to `cx=3` at the `apply` slice, while `h`/`x` elsewhere
  stay as `h`/`x` rather than exploding into internal `u`-gates
- switching the oracle option (0 -> xor) and the step slider both correctly
  update `_resource_stats` and `_current_slice_facts()`
- the coverage scorer gives 1.0 on the hand-written reference and 0.0 on a
  deliberately vague stand-in narration
- confirmed the `apply` slice reports `entangled_qubits: []` for the DJ
  oracle even though it's built from CNOTs — correct: with the ancilla in
  `|->`, phase kickback imparts a phase without entangling x and y, so the
  state stays a product state
- `ollama_client.py` tested against mocked normal / thinking-only / empty
  responses, all three handled without a silent blank
- `compare_backends.py` run end-to-end under plain `python3` (no IPython)
  against mocked Anthropic + Ollama responses, confirming it no longer
  needs a notebook/display context after the `headless_facts.py` rewrite
- all three backends (Anthropic, Ollama, and the OpenAI-compatible
  aggregator) driven live in a running Jupyter widget: a real Ollama
  chat-completion call, a real Anthropic 401 from an invalid credential
  (confirming the request actually reaches Anthropic's API, not just that
  errors are handled), and a full aggregator round trip — URL/key entry,
  live model-list refresh, and a genuine grounded narration back

### Usage

Interactive, in a notebook:
```python
from ai_circuit_slicer import AICircuitSlicer

s = AICircuitSlicer(DeutschJozsa(3),
                     algo_description="Determine whether a boolean oracle "
                                       "is constant or balanced using a "
                                       "single query.")
```
Pick a backend (Anthropic, Ollama, or an OpenAI-compatible aggregator)
from the dropdown, step through slices, click "AI Explain this slice."

Headless, from the command line:
```bash
python compare_backends.py
python compare_backends.py --ollama-model qwen2.5:7b
python compare_backends.py --skip-anthropic   # Ollama only
python compare_backends.py --skip-ollama      # Claude only
```
