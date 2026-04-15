from itertools import chain
from functools import reduce, partial
from collections import defaultdict

import numpy as np
import sympy as sp

from qiskit.quantum_info import Operator, partial_trace, entropy
from qiskit_aer import AerSimulator
from qiskit.visualization import array_to_latex, plot_histogram, plot_bloch_multivector, plot_state_paulivec, plot_state_city
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import clear_output, HTML, Latex, Math

class CircuitSlicer:
    def __init__(self, algo, nsims=1, options=None, common_factors=True, diag=True, matrix_precision=10, decomp=1):
        self.algo = algo
        self.decomp = decomp
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
        self.matrix = widgets.Output(layout=widgets.Layout(width='40%'))
        self.state = widgets.Output(layout=widgets.Layout(width='15%'))
        self.dirac = widgets.Output()
        self.dm = widgets.Output(layout=widgets.Layout(width='45%'))
        self.qinfo = widgets.Output(layout=widgets.Layout(width='20%'))
        self.nsims.observe(self.sim, names='value')
        self.option.observe(self.sim, names='value')
        self.step.observe(self.update, names='value')
        self.mult.observe(self.update, names='value')
        sample = widgets.Button(description="Sample on QPU")
        sampleQasm = widgets.Button(description="Sample on QPU via QASM")
        sample.on_click(partial(self.sample_on_qpu, via_qasm=False))
        sampleQasm.on_click(partial(self.sample_on_qpu, via_qasm=True))
        main = widgets.VBox([widgets.HBox([self.circuit, self.func]),
                             widgets.HBox([widgets.HBox([self.matrix, self.state, self.dm], layout=widgets.Layout(width='80%')),
                                           self.qinfo])])
        self.stats = widgets.Output()
        self.svstats = widgets.Output()
        self.tab = widgets.Tab()
        self.tab.children = [main, widgets.HBox([self.svstats, self.stats])]
        self.tab.set_title(0, "Main")
        self.tab.set_title(1, "Stats")
        self.diag = diag
        if diag:
            self.city = widgets.Output()
            self.tab.children += (widgets.Output(), widgets.Output(), widgets.Output())
            self.tab.set_title(2, "Bloch")
            self.tab.set_title(3, "Pauli")
            self.tab.set_title(4, "City")
        if hasattr(self.algo, "stepproc"):
            self.tab.children += (widgets.Output(),)
            self.tab.set_title(len(self.tab.children)-1, "Step Proc")
        self.out = widgets.VBox([widgets.HBox([self.nsims, self.option, self.step, self.mult]),
                                 widgets.HBox([sample, sampleQasm]),
                                 self.status, self.tab, self.dirac])
        self.sim()
        display(self.out)

    def qbit_index(self, q):
        return self.qc.find_bit(q).index

    def instrument_circuit(self, c):
        def instrument_label(l):
            self.labels.append(l)
            self.instrumented.save_statevector(label=l)
            self.instrumented.save_density_matrix(live, label=l + ":dm")
            self.nonms[l] = sorted(live, key=self.qbit_index)
            try:
                self.ops[l] = Operator(inc)
            except: # a workaround for ops that cannot be converted to instructions
                self.ops[l] = None

        self.ops = {}
        self.nonms = {}
        inc = c.copy_empty_like() # incremental circuit
        self.instrumented = c.copy_empty_like()
        live = set(c.qubits) # non-measured qubits, TODO account for qubits being reset after measurement using density matrix
        self.labels = []
        for i, inst in enumerate(c.data):
            if inst.operation.name == "barrier":
                instrument_label(inst.operation.label)
                inc = c.copy_empty_like()
            else:
                self.instrumented.append(inst.operation, inst.qubits, inst.clbits)
                if inst.operation.name != "measure":
                    # TODO invent something more clever in the presense of measurements
                    inc.append(inst.operation, inst.qubits, inst.clbits)
                else:
                    live -= set(inst.qubits)
        instrument_label("final")
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

    def clear_all(self, o):
        if hasattr(o, "children"):
            for w in o.children:
                if isinstance(w, widgets.Output):
                    w.clear_output(wait=False)
                self.clear_all(w)
    def sim(self, change=None):
        self.res = None
        with self.status:
            try:
                self.qc = self.algo.get_circuit(self.option.value, self.option.label)
                self.instrument_circuit(self.qc)
                self.res = AerSimulator().run(self.instrumented.decompose(reps=self.decomp), shots=self.nsims.value, memory=True).result()
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
                if hasattr(self.algo, "postproc"):
                    display(self.algo.postproc(self.res, self.option.value))
            except Exception as e:
                self.clear_all(self.out)
                get_ipython().showtraceback()
                return
        with self.circuit:
            clear_output(wait=True)
            display(self.qc.draw("mpl"))
        with self.func:
            clear_output(wait=True)
            print("Option:")
            f = self.option.value()
            if hasattr(f, "draw"):
                display(f.draw("mpl"))
            elif isinstance(f, tuple):
                for i in f:
                    display(i.draw("mpl"))
            else:
                display(array_to_latex(f, precision=3))
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
        if hasattr(self.algo, "stepproc"):
            with self.tab.children[-1]:
                clear_output(wait=True)
                display(self.algo.stepproc(sv, dm))
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
        if self.diag:
            with self.tab.children[2]:
                clear_output(wait=True)
                display(plot_bloch_multivector(sv))
            with self.tab.children[3]:
                clear_output(wait=True)
                display(plot_state_paulivec(sv))
            with self.tab.children[4]:
                clear_output(wait=True)
                display(plot_state_city(sv))

    def sample_on_qpu(self, _b, via_qasm=False):
        service = QiskitRuntimeService()
        backend = service.least_busy(simulator=False, operational=True)
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        sampler = SamplerV2(mode=backend)
        try:
            isa_circuit = pm.run(self.qc.decompose(reps=self.decomp))
            # as of April 2026, runtime deserializer doesn't support ancilla registers (https://github.com/Qiskit/qiskit-ibm-runtime/issues/2429)
            # using QASM allows to workaround this without rebuilding the circuit
            if via_qasm:
                from qiskit.qasm3 import dumps, loads
                isa_circuit = loads(dumps(isa_circuit)) 
            job = sampler.run([isa_circuit])
            result = job.result()[0]
            with self.status:
                display(HTML("<b>Sampling results:</b>"))
                for r in result.data.keys():
                    from functools import partial
                    display(HTML(f"<b>{r}</b>: {result.data[r].get_counts()}"))
        except Exception as e:
            self.clear_all(self.out)
            with self.status:
                get_ipython().showtraceback()

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
