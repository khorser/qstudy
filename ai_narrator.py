"""
AI narration for CircuitSlicer slices.

Design principle: the LLM never sees the raw circuit or gets asked to
"figure out" what's happening. It only sees facts CircuitSlicer /
resource_stats.py have already computed (gate counts, entanglement entropy,
purity, non-measured qubits). This is deliberate grounding -- it turns the
LLM into a narrator of verified facts rather than a source of them, which
is the difference between a demo and a tool you'd trust on a real
resource-estimation workflow.
"""
from dataclasses import dataclass, field
from typing import Optional


SYSTEM_PROMPT = (
    "You are a quantum computing teaching assistant embedded in a circuit-"
    "debugging tool. You will be given ONLY facts that have already been "
    "computed by simulation/static analysis for one slice of a circuit "
    "(gate counts, entanglement entropy, purity, algorithm context). "
    "Explain in 2-4 sentences what this slice accomplishes and why, in "
    "plain language suitable for someone who knows linear algebra but is "
    "learning quantum algorithms. Do not invent facts, gate names, or "
    "numbers beyond what is given. If something is not stated, say so "
    "rather than guessing."
)


@dataclass
class SliceFacts:
    algo_name: str
    algo_description: str
    label: str
    gate_counts: dict
    entangling_gates: int
    depth: int
    measurements: int
    purity: float
    entangled_qubits: list
    pairwise_entanglement: dict = field(default_factory=dict)
    cumulative_total_gates: Optional[int] = None


def build_slice_prompt(facts: SliceFacts) -> str:
    gate_str = ", ".join(f"{k}={v}" for k, v in sorted(facts.gate_counts.items())) or "none"
    pair_str = (
        ", ".join(f"q{i}-q{j}: {e:.3f} bits" for (i, j), e in facts.pairwise_entanglement.items())
        or "none above threshold"
    )
    return (
        f"Algorithm: {facts.algo_name}\n"
        f"Context: {facts.algo_description}\n"
        f"Slice label: \"{facts.label}\"\n"
        f"Gates in this slice: {gate_str}\n"
        f"Entangling (>=2-qubit) gate count: {facts.entangling_gates}\n"
        f"Slice depth: {facts.depth}\n"
        f"Measurements in this slice: {facts.measurements}\n"
        f"Density-matrix purity after this slice: {facts.purity:.4f}\n"
        f"Entangled qubits after this slice: {facts.entangled_qubits or 'none'}\n"
        f"Pairwise entanglement entropy: {pair_str}\n"
        + (
            f"Cumulative gate count so far: {facts.cumulative_total_gates}\n"
            if facts.cumulative_total_gates is not None
            else ""
        )
        + "\nExplain what happened in this slice and why, grounded only in the above."
    )


def explain_slice(client, facts: SliceFacts, model="claude-sonnet-5", max_tokens=300) -> str:
    """client: an anthropic.Anthropic() instance, passed in by the caller so
    this module never touches API keys directly."""
    prompt = build_slice_prompt(facts)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------------------
# Lightweight benchmark: hand-written reference explanations for
# Deutsch-Jozsa / Bernstein-Vazirani, plus a coverage scorer. This is not a
# rigorous eval framework -- it's a cheap, defensible first pass at
# "measure against an expert baseline", the kind of thing to extend with a
# real rubric/LLM-judge once there's a second algorithm to compare against.
# ---------------------------------------------------------------------------

DJ_BV_REFERENCE = {
    "init": (
        "Flips the ancilla qubit to |1>, so the next Hadamard puts it into "
        "the |-> state, an eigenstate of X with eigenvalue -1. This sets up "
        "phase kickback: any CNOT-like oracle action on the ancilla will "
        "show up as a phase on the control qubits instead of changing the "
        "ancilla's readout."
    ),
    "prep": (
        "Applies Hadamard to every qubit (input register and ancilla), "
        "creating a uniform superposition over all 2^n input states while "
        "putting the ancilla into the |-> eigenstate prepared in the "
        "previous slice."
    ),
    "apply": (
        "Applies the oracle U_f. Because the ancilla is in the |-> "
        "eigenstate, U_f's CNOTs kick back a phase of (-1)^f(x) onto each "
        "basis state |x> in the input register's superposition, without "
        "entangling the ancilla into the final readout or requiring a "
        "measurement."
    ),
    "done": (
        "Applies Hadamard again to the input register. This is the inverse "
        "of the transform used to build the uniform superposition, so it "
        "converts the phase pattern (-1)^f(x) written onto the state into "
        "amplitude/basis-state information that can be read out directly."
    ),
    "final": (
        "Measures the input register. For Deutsch-Jozsa, an all-zero "
        "result means f is constant and any nonzero result means f is "
        "balanced. For Bernstein-Vazirani, the measured bitstring is "
        "exactly the hidden string s."
    ),
}

_KEY_TERMS = {
    "init": ["ancilla", "eigenstate", ["phase kickback", "kick back a phase", "kick-back"], "-1"],
    "prep": ["superposition", "hadamard", "ancilla"],
    "apply": ["oracle", "phase", ["kickback", "kick back", "kick-back"]],
    "done": ["hadamard", "phase", "amplitude", "basis"],
    "final": ["measure", "constant", "balanced", "string"],
}


def score_coverage(label: str, narration: str) -> dict:
    """Very cheap benchmark: fraction of expected key terms present in the
    AI narration, case-insensitive substring match. Useful as a smoke test
    to flag a narration that's technically fluent but missed the point
    (e.g. never mentions "phase kickback" for the apply slice), not as a
    substitute for expert review.

    Each entry in _KEY_TERMS is either a string or a list of acceptable
    phrasings for the same concept (any one match counts)."""
    terms = _KEY_TERMS.get(label, [])
    if not terms:
        return {"label": label, "coverage": None, "missing": []}
    lower = narration.lower()

    def hit(term):
        variants = term if isinstance(term, list) else [term]
        return any(v.lower() in lower for v in variants)

    missing = [term for term in terms if not hit(term)]
    return {
        "label": label,
        "coverage": (len(terms) - len(missing)) / len(terms),
        "missing": missing,
    }
