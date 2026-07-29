"""
Compare Anthropic (Claude) vs a local Ollama model narrating the same
circuit slices, scored against the hand-written expert-baseline reference
in ai_narrator.DJ_BV_REFERENCE.

This is real, unmocked code meant to run in your own environment with:
  - ANTHROPIC_API_KEY set (for the Anthropic leg)
  - a local Ollama server running with a pulled model (for the Ollama leg)
Either leg is skipped gracefully (with a note in the report) if its backend
isn't reachable, so you can run this with only one of the two configured.

Usage:
    python compare_backends.py
    python compare_backends.py --ollama-model qwen2.5:14b
    python compare_backends.py --skip-anthropic
    python compare_backends.py --skip-ollama
"""
import argparse
import sys

from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit

from ai_circuit_slicer import AICircuitSlicer
from ai_narrator import explain_slice, score_coverage, DJ_BV_REFERENCE


# Copied from fundamentals-of-quantum-algorithms.ipynb. If you've since
# edited the notebook version, keep this in sync (or import it directly
# from a shared module instead of copy-pasting, as a follow-up cleanup).
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

    def get_options(self):
        return [("0", self.f_0), ("xor", self.f_xor)]

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


def build_anthropic_client():
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed"
    try:
        return anthropic.Anthropic(), None
    except Exception as e:
        return None, f"could not create client (check ANTHROPIC_API_KEY): {e}"


def build_ollama_client(base_url):
    try:
        from ollama_client import OllamaClient
        client = OllamaClient(base_url=base_url)
        return client, None
    except Exception as e:
        return None, f"could not create client: {e}"


def run_backend(name, client, model, slicer, labels):
    """Returns {label: {"narration": str, "coverage": float|None, "missing": [...]}}
    or None if the client itself failed to build."""
    if client is None:
        return None
    rows = {}
    for label in labels:
        slicer.step.value = label
        facts = slicer._current_slice_facts()
        try:
            narration = explain_slice(client, facts, model=model)
        except Exception as e:
            narration = f"[ERROR calling {name}: {e}]"
        scored = score_coverage(label, narration)
        rows[label] = {
            "narration": narration,
            "coverage": scored["coverage"],
            "missing": scored["missing"],
        }
    return rows


def print_report(labels, anthropic_rows, ollama_rows):
    print("=" * 100)
    print(f"{'Slice':<10} {'Claude coverage':<18} {'Ollama coverage':<18}")
    print("-" * 100)
    for label in labels:
        a_cov = anthropic_rows[label]["coverage"] if anthropic_rows else None
        o_cov = ollama_rows[label]["coverage"] if ollama_rows else None
        a_str = f"{a_cov:.2f}" if a_cov is not None else "skipped"
        o_str = f"{o_cov:.2f}" if o_cov is not None else "skipped"
        print(f"{label:<10} {a_str:<18} {o_str:<18}")
    print("=" * 100)

    for label in labels:
        print(f"\n--- Slice: {label} ---")
        print(f"Reference : {DJ_BV_REFERENCE[label]}")
        if anthropic_rows:
            r = anthropic_rows[label]
            print(f"Claude    : {r['narration']}")
            if r["missing"]:
                print(f"  (missing expected concepts: {r['missing']})")
        if ollama_rows:
            r = ollama_rows[label]
            print(f"Ollama    : {r['narration']}")
            if r["missing"]:
                print(f"  (missing expected concepts: {r['missing']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anthropic-model", default="claude-sonnet-5")
    parser.add_argument("--ollama-model", default="llama3.1")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--skip-anthropic", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    args = parser.parse_args()

    slicer = AICircuitSlicer(
        DeutschJozsa(3),
        diag=False,
        common_factors=False,
        algo_description=DeutschJozsa.__doc__,
    )
    # We call explain_slice() directly per backend below rather than going
    # through the widget's single _build_client()/dropdown, since we want
    # both backends' results side by side rather than whichever one the
    # dropdown happens to be set to.
    slicer.option.value = slicer.option.options[1][1]  # "xor" oracle

    labels = list(slicer.step.options)

    anthropic_rows = None
    if not args.skip_anthropic:
        client, err = build_anthropic_client()
        if err:
            print(f"[Anthropic skipped: {err}]", file=sys.stderr)
        else:
            anthropic_rows = run_backend("Anthropic", client, args.anthropic_model, slicer, labels)

    ollama_rows = None
    if not args.skip_ollama:
        client, err = build_ollama_client(args.ollama_url)
        if err:
            print(f"[Ollama skipped: {err}]", file=sys.stderr)
        else:
            ollama_rows = run_backend("Ollama", client, args.ollama_model, slicer, labels)

    if anthropic_rows is None and ollama_rows is None:
        print("Neither backend was available -- nothing to compare.", file=sys.stderr)
        sys.exit(1)

    print_report(labels, anthropic_rows, ollama_rows)


if __name__ == "__main__":
    main()
