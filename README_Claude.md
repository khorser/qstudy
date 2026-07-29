# AI-Assisted CircuitSlicer: Resource Estimation + Grounded Narration

An additive extension to `qstudy.py`'s `CircuitSlicer` — an interactive,
barrier-delimited circuit-stepping tool already used across your
Deutsch-Jozsa, Bernstein-Vazirani, Simon, Grover, and teleportation demos.
`qstudy.py` is untouched; everything here is a subclass + standalone
analysis modules layered on top.

## What's new

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
purity, depth) and asks Claude to narrate what a slice does and why. The
LLM never sees the raw circuit and is explicitly told not to invent
numbers beyond what's given — grounding by construction, not by
instruction alone. Includes a small hand-written reference (expert
baseline) for the Deutsch-Jozsa/Bernstein-Vazirani slices and a coverage
scorer, tested to confirm it scores the reference itself at 1.0 and a
deliberately vague narration at 0.0 — a cheap first pass at "benchmark
against an expert baseline," worth extending once there's a second
algorithm to compare against.

**`ai_circuit_slicer.py`** — `AICircuitSlicer(CircuitSlicer)`: adds a
"Resources" tab (rendered resource_stats table) and an "AI Explain this
slice" button wired to ai_narrator. Degrades gracefully with no API key
configured (shows the prompt that would have been sent, rather than
failing to build).

## Verified, not just written

Ran end-to-end under an IPython kernel context against a reconstructed
`DeutschJozsa` class from your notebook:
- oracle-gate resolution: `f_xor` oracle correctly resolves from an opaque
  `circuit-N` blob to `cx=3` at the `apply` slice, while `h`/`x` elsewhere
  stay as `h`/`x` rather than exploding into internal `u`-gates
- switching the oracle option (0 -> xor) and the step slider both correctly
  update `_resource_stats` and `_current_slice_facts()`
- the coverage scorer gives 1.0 on the hand-written reference and 0.0 on a
  deliberately vague stand-in narration, confirming it actually
  discriminates rather than just returning a fixed score
- confirmed the `apply` slice reports `entangled_qubits: []` for the DJ
  oracle even though it's built from CNOTs — correct: with the ancilla in
  `|->`, phase kickback imparts a phase without entangling x and y, so the
  state stays a product state. Worth noting in narration rather than
  assuming CNOTs always entangle.
