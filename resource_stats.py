"""
Static resource-estimation pass over a barrier-delimited circuit.

Deliberately independent of CircuitSlicer's simulation path: it re-walks
circuit.data using the same barrier-segmentation convention (so labels line
up with CircuitSlicer's slices), but does no simulation of its own. This
keeps the existing, working instrumentation logic in qstudy.py untouched.
"""
from collections import Counter

# Gates qiskit would otherwise expand into a chain of internal `u`/`cx`
# during a blanket .decompose() call. We leave these alone and only unroll
# instructions *outside* this set (custom oracles, black-box subroutines),
# so counts stay both accurate (black boxes resolved) and readable (h stays
# "h", not 3 u-gates).
_PRIMITIVE_GATES = {
    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
    "cx", "cy", "cz", "ch", "swap", "ccx", "cswap",
    "crx", "cry", "crz", "cp", "cu", "id",
}


def _resolve_opaque_gates(circ, max_reps=6):
    """Repeatedly decompose only non-primitive (opaque/custom) instructions
    until none remain or max_reps is hit. Standard gates are left intact."""
    for _ in range(max_reps):
        names = {inst.operation.name for inst in circ.data}
        opaque = names - _PRIMITIVE_GATES
        if not opaque:
            break
        circ = circ.decompose(gates_to_decompose=list(opaque))
    return circ


def slice_resource_stats(circuit, resolve_opaque=True):
    """
    Returns an ordered dict: {label: {"slice": {...}, "cumulative": {...}}}

    Per slice / cumulative metrics:
      - gate_counts: dict of op-name -> count (barriers and measures excluded
        from gate_counts; measures tracked separately)
      - total_gates: sum of gate_counts
      - entangling_gates: count of instructions acting on >=2 qubits
      - measurements: count of measure ops in that slice
      - depth: circuit depth of just that slice's ops (slice-local only,
        not meaningful/omitted for cumulative)

    resolve_opaque: if True (default), custom/opaque gates (e.g. an oracle
    appended via .append(f().to_gate())) are unrolled to primitive gates
    before counting, so they don't show up as a single opaque "circuit-N"
    op that hides the real cost. Standard gates (h, x, cx, ...) are left
    as-is either way.
    """
    segments = []
    seg = circuit.copy_empty_like()
    seg_measure_count = 0
    for inst in circuit.data:
        if inst.operation.name == "barrier":
            segments.append((inst.operation.label, seg, seg_measure_count))
            seg = circuit.copy_empty_like()
            seg_measure_count = 0
        elif inst.operation.name == "measure":
            seg_measure_count += 1
        else:
            seg.append(inst.operation, inst.qubits, inst.clbits)
    segments.append(("final", seg, seg_measure_count))

    results = {}
    cum_counts = Counter()
    cum_entangling = 0
    cum_measure = 0
    for label, seg_circ, measure_count in segments:
        if resolve_opaque:
            seg_circ = _resolve_opaque_gates(seg_circ)
        counts = Counter(seg_circ.count_ops())
        entangling = sum(
            1 for inst in seg_circ.data if len(inst.qubits) >= 2
        )
        depth = seg_circ.depth()

        cum_counts += counts
        cum_entangling += entangling
        cum_measure += measure_count

        results[label] = {
            "slice": {
                "gate_counts": dict(counts),
                "total_gates": sum(counts.values()),
                "entangling_gates": entangling,
                "measurements": measure_count,
                "depth": depth,
            },
            "cumulative": {
                "gate_counts": dict(cum_counts),
                "total_gates": sum(cum_counts.values()),
                "entangling_gates": cum_entangling,
                "measurements": cum_measure,
            },
        }
    return results


def render_resource_html(stats):
    """Render slice_resource_stats() output as an HTML table for display in
    a widgets.Output()."""
    rows = []
    for label, s in stats.items():
        sl, cu = s["slice"], s["cumulative"]
        gate_str = ", ".join(f"{k}:{v}" for k, v in sorted(sl["gate_counts"].items())) or "—"
        rows.append(
            f"<tr><td><b>{label}</b></td>"
            f"<td>{gate_str}</td>"
            f"<td>{sl['total_gates']}</td>"
            f"<td>{sl['entangling_gates']}</td>"
            f"<td>{sl['depth']}</td>"
            f"<td>{sl['measurements']}</td>"
            f"<td>{cu['total_gates']}</td>"
            f"<td>{cu['entangling_gates']}</td></tr>"
        )
    header = (
        "<tr><th>Slice</th><th>Gates (this slice)</th><th>Total</th>"
        "<th>Entangling</th><th>Depth</th><th>Measure</th>"
        "<th>Cum. total</th><th>Cum. entangling</th></tr>"
    )
    return f"<table border=1 cellpadding=4>{header}{''.join(rows)}</table>"
