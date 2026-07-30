"""
Headless equivalent of CircuitSlicer's simulation + instrumentation --
no ipywidgets, no display(), no IPython dependency at all. Produces the
same per-slice facts (SliceFacts) that AICircuitSlicer's "AI Explain"
button uses, for scripts like compare_backends.py that need them without
building the interactive widget (which requires display() and therefore a
real IPython/Jupyter frontend).

Mirrors CircuitSlicer.instrument_circuit()/sim() in qstudy.py in terms of
slice semantics (barrier-delimited segments, "final" as the last implicit
label) -- deliberately not importing anything from qstudy.py, since that
module pulls in ipywidgets at import time.
"""
from qiskit.quantum_info import partial_trace, entropy
from qiskit_aer import AerSimulator

from resource_stats import slice_resource_stats
from ai_narrator import SliceFacts


def _qbit_index(qc, q):
    return qc.find_bit(q).index


def instrument_circuit_headless(qc):
    """Returns (instrumented_circuit, labels), matching the barrier
    segmentation CircuitSlicer.instrument_circuit uses, but with no widget
    state and no Operator/incremental-circuit bookkeeping (that's only
    needed for the interactive matrix-per-slice widget, not for facts)."""
    instrumented = qc.copy_empty_like()
    live = set(qc.qubits)
    labels = []

    def save_slice(label):
        labels.append(label)
        instrumented.save_statevector(label=label)
        instrumented.save_density_matrix(
            sorted(live, key=lambda q: _qbit_index(qc, q)), label=label + ":dm"
        )

    for inst in qc.data:
        if inst.operation.name == "barrier":
            save_slice(inst.operation.label)
        else:
            instrumented.append(inst.operation, inst.qubits, inst.clbits)
            if inst.operation.name == "measure":
                live -= set(inst.qubits)
    save_slice("final")
    return instrumented, labels


def compute_all_slice_facts(qc, algo_name, algo_description, shots=1, decompose_reps=1):
    """Runs the circuit once and returns {label: SliceFacts} for every
    barrier-delimited slice, reusing the same resource_stats pass
    AICircuitSlicer's Resources tab uses. No display/ipywidgets involved --
    safe to call from a plain script or CI job."""
    instrumented, labels = instrument_circuit_headless(qc)
    result = AerSimulator().run(
        instrumented.decompose(reps=decompose_reps), shots=shots, memory=True
    ).result()

    stats = slice_resource_stats(qc)

    facts = {}
    for label in labels:
        dm = result.data()[label + ":dm"]
        num_q = dm.num_qubits
        entangled = [
            i for i in range(num_q)
            if entropy(partial_trace(dm, [j for j in range(num_q) if j != i])) > 1e-10
        ]
        pairwise = {}
        for i in range(num_q):
            for j in range(i + 1, num_q):
                trace_out = [k for k in range(num_q) if k not in (i, j)]
                e = entropy(partial_trace(dm, trace_out), base=2)
                if e > 1e-10:
                    pairwise[(i, j)] = e

        slice_stats = stats[label]["slice"]
        cumulative_stats = stats[label]["cumulative"]
        facts[label] = SliceFacts(
            algo_name=algo_name,
            algo_description=algo_description,
            label=label,
            gate_counts=slice_stats["gate_counts"],
            entangling_gates=slice_stats["entangling_gates"],
            depth=slice_stats["depth"],
            measurements=slice_stats["measurements"],
            purity=dm.purity().real,
            entangled_qubits=entangled,
            pairwise_entanglement=pairwise,
            cumulative_total_gates=cumulative_stats["total_gates"],
        )
    return facts
