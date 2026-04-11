from itertools import chain
from functools import reduce
from collections import defaultdict

import numpy as np
import sympy as sp

from qiskit.quantum_info import Operator, partial_trace, entropy
from qiskit_aer import AerSimulator
from qiskit.visualization import array_to_latex, plot_histogram, plot_bloch_multivector, plot_state_paulivec, plot_state_city

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import clear_output, HTML, Latex, Math

class CircuitSlicer:
    def __init__(self, algo, nsims=1, options=None, common_factors=True, diag=True, postproc=None, matrix_precision=10, stepproc=None):
        self.algo = algo
        self.matrix_precision = matrix_precision
        if options is None:
            options = algo.get_options()
        self.nsims = widgets.IntText(nsims, description="# of sims: ")
        self.option = widgets.Dropdown(options=options, description="Option: ")
        self.step = widgets.SelectionSlider(options=[""], description="Step: ")
        self.mult = widgets.Checkbox(common_factors, description="Extract common factors")
        self.status = widgets.Output()
        self.func = widgets.Output(layout=widgets.Layout(width='20%'))
        self.circuit = widgets.Output(layout=widgets.Layout(width='80%'))
        self.bloch = diag and widgets.Output()
        self.pauli = diag and widgets.Output()
        self.city = diag and widgets.Output()
        self.matrix = widgets.Output(layout=widgets.Layout(width='40%'))
        self.state = widgets.Output(layout=widgets.Layout(width='15%'))
        self.dirac = widgets.Output()
        self.dm = widgets.Output(layout=widgets.Layout(width='45%'))
        self.qinfo = widgets.Output(layout=widgets.Layout(width='20%'))
        self.nsims.observe(self.sim, names='value')
        self.option.observe(self.sim, names='value')
        self.step.observe(self.update, names='value')
        self.mult.observe(self.update, names='value')
        main = widgets.VBox([widgets.HBox([self.circuit, self.func]),
                                  widgets.HBox([widgets.HBox([self.matrix, self.state, self.dm], layout=widgets.Layout(width='80%')),
                                                self.qinfo])])
        self.stats = widgets.Output()
        self.svstats = widgets.Output()
        tab = widgets.Tab()
        tab.children = [main, widgets.HBox([self.svstats, self.stats])]
        tab.set_title(0, "Main")
        tab.set_title(1, "Stats")
        if diag:
            tab.children += (self.bloch, self.pauli, self.city)
            tab.set_title(2, "Bloch")
            tab.set_title(3, "Pauli")
            tab.set_title(4, "City")
        self.out = widgets.VBox([widgets.HBox([self.nsims, self.option, self.step, self.mult]), self.status, tab, self.dirac])
        self.postproc = postproc
        self.stepproc = stepproc
        self.sim()
        display(self.out)

    def qbit_index(self, q):
        return self.qc.find_bit(q).index

    def instrument_circuit(self, c):
        def instrument_label(qc, l):
            self.labels.append(l)
            qc.save_statevector(label=l)
            qc.save_density_matrix(live, label=l + ":dm")
            self.nonms[l] = sorted(live, key=self.qbit_index)
            try:
                self.ops[l] = Operator(inc)
            except: # a workaround for ops that cannot be converted to instructions
                self.ops[l] = None

        self.ops = {}
        self.nonms = {}
        inc = c.copy_empty_like() # incremental circuit
        new = c.copy_empty_like()
        live = set(c.qubits) # non-measured qubits, TODO account for qubits being reset after measurement using density matrix
        self.labels = []
        for i, inst in enumerate(c.data):
            if inst.operation.name == "barrier":
                instrument_label(new, inst.operation.label)
                inc = c.copy_empty_like()
            else:
                new.append(inst.operation, inst.qubits, inst.clbits)
                if inst.operation.name != "measure":
                    # TODO invent something more clever in the presense of measurements
                    inc.append(inst.operation, inst.qubits, inst.clbits)
                else:
                    live -= set(inst.qubits)
        instrument_label(new, "final")
        try:
            self.labels.index(self.step.value)
            newl = self.step.value
        except:
            newl = self.labels[0]
            for a, b in zip(self.labels, self.step.options):
                if a == b:
                    newl = a
                else:
                    break
        self.step.options = self.labels
        self.value = newl
        return new

    def sim(self, change=None):
        def clear_all(o):
            if hasattr(o, 'children'):
                for w in o.children:
                    if isinstance(w, widgets.Output):
                        w.clear_output(wait=False)
                    clear_all(w)
        self.res = None
        with self.status:
            try:
                self.qc = self.algo.get_circuit(self.option.value, self.option.label)
                nc = self.instrument_circuit(self.qc)
                self.res = AerSimulator().run(nc.decompose(), shots=self.nsims.value, memory=True).result()
                clear_output(wait=True)
                def get_clbits(a, x):
                    pos, prev, txt = a
                    b = self.qc.find_bit(x)
                    n = b.registers[0][0].name
                    if n != prev:
                        sep = r"{\quad}"
                        pos = 0
                    else:
                        sep = ""
                    txt.append([b.registers[0][0].name, "^{(", str(b.index), ")}_{", str(pos), "}", sep])
                    return (pos+1, n, txt)
                _, _, txt = reduce(get_clbits, self.qc.clbits, (0, "", []))
                display(Latex(f"Classical Registers: ${''.join(chain.from_iterable(reversed(txt)))}$"))
                display(HTML(f"<b>Measurements: {self.res.get_memory()}</b>"))
                if self.postproc is not None:
                    display(self.postproc(self.res, self.option.value))
            except Exception as e:
                clear_all(self.out)
                get_ipython().showtraceback()
                return
        with self.circuit:
            clear_output(wait=True)
            display(self.qc.draw("mpl"))
        with self.func:
            clear_output(wait=True)
            print("Option:")
            try:
                display(self.option.value().draw("mpl"))
            except:
                display(array_to_latex(self.option.value(), precision=3))
        with self.stats:
            clear_output(wait=True)
            display(plot_histogram(self.res.get_counts()))
        self.update()

    def update(self, change=None):
        def draw_matrix(x, name):
            prefix = f"{name} = "
            if self.mult.value:
                return factor_out(x, prefix=prefix)
            else:
                return array_to_latex(x, prefix=prefix, precision=self.matrix_precision)
        if self.res is None:
            return
        l = self.step.value
        sv = self.res.data()[l]
        dm = self.res.data()[l + ":dm"]
        if self.stepproc is not None:
            self.stepproc(self.out, sv, dm)
        with self.matrix:
            clear_output(wait=True)
            if self.ops[l] is None:
                display(HTML("Cannot produce the matrix"))
            else:
                display(draw_matrix(self.ops[l], r"{\cal O}"))
        with self.dirac:
            clear_output(wait=True)
            display(sv.draw("latex"))
        with self.state:
            clear_output(wait=True)
            display(draw_matrix(sv.data.reshape((-1,1)), r"\psi"))
        with self.dm:
            clear_output(wait=True)
            display(draw_matrix(dm, r"\rho"))
        with self.svstats:
            clear_output(wait=True)
            display(plot_histogram(sv.probabilities_dict()))
        with self.qinfo:
            clear_output(wait=True)
            print(f"Purity: {dm.purity():.3f}")
            print("Non measured:")
            for q in self.nonms[l]:
                bit = self.qc.find_bit(q)
                # CAVEAT a Qubit might be a part of multiple registers
                reg_name = bit.registers[0][0].name if bit.registers else "?"
                print(f"{reg_name}[{bit.index}]")
            print("Entangled qubits:")
            print([i for i in range(dm.num_qubits)
                   if entropy(partial_trace(dm, list(chain(range(i), range(i+1, dm.num_qubits))))) > 1e-10])
            print("Pair-wise entanglements:")
            for i in range(dm.num_qubits):
                for j in range(i+1, dm.num_qubits):
                    keep = [i, j]
                    trace_out = [k for k in range(dm.num_qubits) if k not in keep]
                    reduced = partial_trace(dm, trace_out)
                    e = entropy(reduced, base=2)
                    if e > 1e-10:
                        print(f"q{i}-q{j}: {e:.3f}")
        if self.bloch:
            with self.bloch:
                clear_output(wait=True)
                display(plot_bloch_multivector(sv))
        if self.pauli:
            with self.pauli:
                clear_output(wait=True)
                display(plot_state_paulivec(sv))
        if self.city:
            with self.city:
                clear_output(wait=True)
                display(plot_state_city(sv))

def factor_out(arr, prefix=""):
# generated with Gemini and Claude
    arr = np.asarray(arr, dtype=complex)
    shape = arr.shape

    def to_sympy(z):
        re_f = round(z.real, 12)
        im_f = round(z.imag, 12)

        irrational_hints = [sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)]
        re = sp.nsimplify(re_f, irrational_hints, rational=False)
        im = sp.nsimplify(im_f, irrational_hints, rational=False)
        if abs(float(re) - re_f) > 1e-8:
            re = sp.nsimplify(re_f, rational=True)
        if abs(float(im) - im_f) > 1e-8:
            im = sp.nsimplify(im_f, rational=True)
        return re + sp.I * im

    sym_flat = [to_sympy(z) for z in arr.flatten()]

    parts = []
    for z in sym_flat:
        re, im = z.as_real_imag()
        if re != 0: parts.append(sp.Abs(re))
        if im != 0: parts.append(sp.Abs(im))

    if not parts:
        display(Math(f"{name} = {sp.latex(sp.zeros(*shape))}"))
        return

    def split_rational(expr):
        """split expr into (rational_coeff, irrational_unit).
        e.g. sqrt(3)/2 -> (1/2, sqrt(3)), 1/2 -> (1/2, 1), sqrt(2) -> (1, sqrt(2))
        """
        expr = sp.nsimplify(expr)
        coeff, rest = expr.as_coeff_Mul()
        return coeff, rest  # coeff ∈ ℚ, rest is irrational

    # group by irrational unit: {unit: [coeff1, coeff2, ...]}
    groups = defaultdict(list)
    for p in parts:
        coeff, unit = split_rational(p)
        groups[unit].append(coeff)

    # GCD of rational coefficients inside each group
    def rational_gcd(a, b):
        # gcd(p/q, r/s) = gcd(p,r) / lcm(q,s)
        a, b = sp.Rational(a), sp.Rational(b)
        return sp.Rational(sp.gcd(a.p, b.p), sp.lcm(a.q, b.q))

    # common multiplier = min by groups of gcd(coefficients) * unit,
    # but there is nothing common between groups - taking gcd for all parts by ratios
    # common multiplier = gcd of all rational_coeff * gcd of all units
    all_coeffs = [c for coeffs in groups.values() for c in coeffs]
    all_units = list(groups.keys())

    rat_common = reduce(rational_gcd, all_coeffs)
    unit_common = reduce(sp.gcd, all_units)  # gcd(sqrt(3), 1) = 1, gcd(1,1) = 1

    common = rat_common * unit_common

    scaled = [sp.radsimp(z / common) for z in sym_flat]
    mat = sp.Matrix(shape[0], shape[1] if len(shape) > 1 else 1, scaled)

    factor_latex = sp.latex(common) if common != 1 else ""
    return Math(rf"{prefix} {factor_latex} {sp.latex(mat)}")
