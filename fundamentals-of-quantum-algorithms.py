# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from collections import Counter
from functools import partial

from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit, __version__ as qiskitver
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.circuit.library import UnitaryGate, UGate
from qiskit.visualization import array_to_latex

import numpy as np
import sympy as sp

from IPython.display import HTML, Latex
import matplotlib.patches as patches
from matplotlib.figure import Figure

from qstudy import CircuitSlicer
qiskitver


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Entanglement in Action

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Teleportation

# %%
class Teleportation:
    def random_q(self):
        return self.u
    def get_options(self):
        return [("Random Q", self.random_q)]
    def get_circuit(self, f, label=""):
        self.u = UGate(
            theta=np.random.random() * 2 * np.pi,
            phi=np.random.random() * 2 * np.pi,
            lam=np.random.random() * 2 * np.pi).to_matrix()
        q = QuantumRegister(1, "Q")
        qa = QuantumRegister(1, "A")
        qb = QuantumRegister(1, "B")
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(1, "b")
        r = ClassicalRegister(1, "r")
        c = QuantumCircuit(q, qa, qb, a, b, r)
        g = UnitaryGate(self.u, label="$U$")
        c.append(g, q)
        c.barrier(label="init")
        c.h(qa)
        c.cx(qa, qb)
        c.barrier(label="prep")
        c.cx(q, qa)
        c.h(q)
        c.barrier(label="A")
        c.measure(qa, a)
        c.measure(q, b)
        with c.if_test((a, 1)):
            c.x(qb)
        with c.if_test((b, 1)):
            c.z(qb)
        c.barrier(label="received")
        ui = UnitaryGate(g.inverse(), label="$U^{-1}$")
        c.append(ui, qb)
        c.measure(qb, r)
        return c


# %%
CircuitSlicer(Teleportation(), common_factors=False, matrix_precision=3);


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Superdence Coding

# %%
class Coding:
    def f00(self):
        return QuantumCircuit(2)
    def f01(self):
        c = self.f00()
        c.x(0)
        return c
    def f10(self):
        c = self.f00()
        c.x(1)
        return c
    def f11(self):
        c = self.f00()
        c.x([0, 1])
        return c
    def get_options(self):
        return [("00", self.f00), ("01", self.f01), ("10", self.f10), ("11", self.f11)]
    def get_circuit(self, f, label=""):
        tmp = QuantumRegister(2, "t")
        qa = QuantumRegister(1, "A")
        qb = QuantumRegister(1, "B")
        a1 = ClassicalRegister(1, "a1")
        a2 = ClassicalRegister(1, "a2")
        b1 = ClassicalRegister(1, "b1")
        b2 = ClassicalRegister(1, "b2")
        c = QuantumCircuit(tmp, qa, qb, a1, a2, b1, b2)
        c.append(f().to_gate(label="Init"), tmp)
        c.measure(tmp[0], a1)
        c.measure(tmp[1], a2)
        c.barrier(label="init")
        c.h(qa)
        c.cx(qa, qb)
        c.barrier(label="prep")
        with c.if_test((a1, 1), label="Z"):
            c.z(qa)
        with c.if_test((a2, 1), label="X"):
            c.x(qa)
        c.barrier(label="apply")

        c.cx(qa, qb)
        c.h(qa)
        c.barrier(label="B")

        b1 = ClassicalRegister(1, "b1")
        b2 = ClassicalRegister(1, "b2")
        c.measure(qa, b1)
        c.measure(qb, b2)
        return c


# %%
CircuitSlicer(Coding());


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## CHSH Game

# %%
class CHSH:
    def __init__(self, aangles, bangles):
        self.aangles, self.bangles = aangles, bangles
    def f00(self):
        return [0, 0]
    def f01(self):
        return [0, 1]
    def f10(self):
        return [1, 0]
    def f11(self):
        return [1, 1]
    def get_options(self):
        return [("00", self.f00), ("01", self.f01), ("10", self.f10), ("11", self.f11)]
    def get_circuit(self, f, label=""):
        qa = QuantumRegister(1, "A")
        qb = QuantumRegister(1, "B")
        x, y = f()
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(1, "b")
        c = QuantumCircuit(qa, qb, a, b)
        c.h(qa)
        c.cx(qa, qb)
        c.barrier(label="prep")
        if x:
            c.ry(self.aangles[0], qa)
        else:
            c.ry(self.aangles[1], qa)
        c.measure(qa, a)
        c.barrier(label="A")
        if y:
            c.ry(self.bangles[0], qb)
        else:
            c.ry(self.bangles[1], qb)
        c.measure(qb, b)
        return c
    def postproc(self, result, option):
        def success(t):
            b, a = t.split(" ")
            x, y = option()
            return (a!=b) == int(x)&int(y)
        c = Counter(map(success, result.get_memory()))
        return HTML(f"<b>Success count:</b> {c[True]}")


# %%
CircuitSlicer(CHSH((-np.pi/2, 0), (np.pi/4, -np.pi/4)), nsims=10);


# %%
class CHSH_Phase:
    def fp4(self):
        return [np.pi/4, -np.pi/4]
    def fp8(self):
        return [np.pi/8, -3*np.pi/8]
    def fp16(self):
        return [np.pi/16, -7*np.pi/16]
    def get_options(self):
        return [("π/4, -π/4", self.fp4), ("π/8, -3*π/8", self.fp8), ("π/16, -7*π/16", self.fp16)]
    def get_circuit(self, f, label=""):
        rnd = QuantumRegister(1, "rnd")
        qa = QuantumRegister(1, "A")
        qb = QuantumRegister(1, "B")
        x = ClassicalRegister(1, "x")
        y = ClassicalRegister(1, "y")
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(1, "b")
        c = QuantumCircuit(rnd, qa, qb, x, y, a, b)
        c.h(rnd)
        c.measure(rnd, x)
        c.h(rnd)
        c.measure(rnd, y)
        c.barrier(label="init")
        c.h(qa)
        c.cx(qa, qb)
        c.barrier(label="prep")
        with c.if_test((x, 1)) as else_:
            c.ry(-np.pi/2, qa)
        with else_:
            c.ry(0, qa)
        c.measure(qa, a)
        c.barrier(label="A")
        with c.if_test((y, 1)) as else_:
            c.ry(f()[0], qb)
        with else_:
            c.ry(f()[1], qb)
        c.measure(qb, b)
        return c
    def postproc(self, result, option):
        def success(t):
            b, a, y, x = t.split(" ")
            return (a!=b) == int(x)&int(y)
        c = Counter(map(success, result.get_memory()))
        return HTML(f"<b>Success count:</b> {c[True]}")


# %%
CircuitSlicer(CHSH_Phase(), nsims=10);


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Deutsch et al

# %% [markdown]
#
# | function | $\ket y \ket x$ | $f(x)$ | $\ket{y \oplus f(x)}\ket x$ | U | formula |
# | -- | -- | -- | -- | -- | -- |
# | **0** | 00 | 0| 00 | 1 0 0 0 | $y$
# | | 01 | 0 | 01 | 0 1 0 0
# | | 10 | 0 | 10 | 0 0 1 0
# | | 11 | 0 | 11 | 0 0 0 1
# | **id** | 00 | 0 | 00 | 1 0 0 0 | CX $y$
# | | 01 | 1 | 11 | 0 0 0 1
# | | 10 | 0 | 10 | 0 0 1 0
# | | 11 | 1 | 01 | 0 1 0 0
# | **not** | 00 | 1 | 10 | 0 0 1 0 | CX X $y$ == X CX $y$
# | | 01 | 0 | 01 | 0 1 0 0
# | | 10 | 1 | 00 | 1 0 0 0
# | | 11 | 0 | 11 | 0 0 0 1
# | **1** | 00 | 1 | 10 | 0 0 1 0 | X $y$
# | | 01 | 1 | 11 | 0 0 0 1
# | | 10 | 1 | 00 | 1 0 0 0
# | | 11 | 1 | 01 | 0 1 0 0

# %%
class Deutsch:
    def f_0(self):
        return QuantumCircuit(2)
    def f_id(self):
        c = self.f_0()
        c.cx(0, 1)
        return c
    def f_not(self):
        c = self.f_0()
        c.cx(0, 1)
        c.x(1)
        return c
    def f_1(self):
        c = self.f_0()
        c.x(1)
        return c
    def get_options(self):
        return [("0", self.f_0), ("id", self.f_id), ("not", self.f_not), ("1", self.f_1)]
    # if you know what are you doing you can pass any function here
    def get_circuit(self, f, label=""):
        x = QuantumRegister(1, name="x")
        y = AncillaRegister(1, name="y")
        r = ClassicalRegister(1, name="r")
        c = QuantumCircuit(x, y, r)
        c.x(y)
        c.barrier(label="init")
        c.h(x)
        c.h(y)
        c.barrier(label="prep")
        c.append(f().to_gate(label=f"$U_{{{label}}}$"), [x, y])
        c.barrier(label="apply")
        c.h(x)
        c.barrier(label="done")
        c.measure(x, r)
        return c


# %%
CircuitSlicer(Deutsch());


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Deutsch-Jozsa

# %%
class DeutschJozsa:
    def __init__(self, n):
        self.n = n
    def f_0(self):
        return QuantumCircuit(self.n+1)
    def f_xor(self):
        c = self.f_0()
        c.cx(list(range(self.n)), self.n)
        return c
    def f_not0th(self):
        c = self.f_0()
        c.cx(0, self.n)
        c.x(self.n)
        return c
    def f_i0n1(self):
        c = self.f_0()
        c.cx([0, 1], self.n)
        c.x(self.n)
        return c
    def f_1(self):
        c = self.f_0()
        c.x(self.n)
        return c
    def f_random(self):
        c = self.f_0()
        # generate binary representation for numbers 0..(n-1)
        all_states = (np.arange(2**self.n)[:, None] >> np.arange(self.n)) & 1
        # to ensure balance, take only a half
        on_states_bits = np.random.permutation(all_states)[:2**(self.n-1)]
        for bits in on_states_bits:
            zeros = np.arange(self.n)[bits == 0].tolist()
            if zeros:
                c.x(zeros)
            c.mcx(list(range(self.n)), self.n)
            if zeros:
                c.x(zeros)
        return c
    def get_options(self):
        return [("0", self.f_0), ("xor", self.f_xor), ("0:not", self.f_not0th), ("0:id, 1:not", self.f_i0n1),
                ("1", self.f_1), ("random balanced", self.f_random)]
    # if you know what are you doing you can pass any function here
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


# %%
CircuitSlicer(DeutschJozsa(3));


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Bernstein-Vazirani

# %%
class BernsteinVazirani (DeutschJozsa):
    def __init__(self, x):
        super().__init__(x.bit_length())
        self.x = x
    def bv(self):
        c = QuantumCircuit(self.n+1)
        c.cx([i for i in range(self.n) if (self.x>>i)&1], self.n)
        return c
    def get_options(self):
        return [("BV", self.bv)]
    def postproc(self, result, option):
        r = result.get_memory()[0]
        return HTML(f"<b>Processing result:</b> s = {int(r, 2)}")
CircuitSlicer(BernsteinVazirani(95), diag=False, common_factors=False);


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Simon

# %% [markdown]
# $$
# \begin{aligned}
# f(000) & = 10011 \\
# f(001) & = 00101 \\
# f(010) & = 00101 \\
# f(011) & = 10011 \\
# f(100) & = 11010 \\
# f(101) & = 00001 \\
# f(110) & = 00001 \\
# f(111) & = 11010
# \end{aligned}
# $$
#
# $F = \left(f(100), f(010), f(001)\right)$ on $GF(2)$ (columns=images of basis vectors, multiplication = AND, addition = XOR):
# $$F =
# \begin{pmatrix}
# 1 & 0 & 0 \\
# 1 & 0 & 0 \\
# 0 & 1 & 1 \\
# 1 & 0 & 0 \\
# 0 & 1 & 1
# \end{pmatrix}
# $$

# %%
class Simon:
    def __init__(self, i, o):
        self.i = i
        self.o = o

    def fixed_ibm_example(self):
        c = QuantumCircuit(8)
        c.cx(0, [3, 4, 6])
        c.cx(1, [5, 7])
        c.cx(2, [5, 7])
        return c

    def get_options(self):
        return [("ibm", self.fixed_ibm_example)]

    def get_circuit(self, f, label=""):
        x = QuantumRegister(self.i, name="x")
        y = AncillaRegister(self.o, name="y")
        r = ClassicalRegister(self.i, name="r")
        tmp = ClassicalRegister(self.o, name="t")
        c = QuantumCircuit(x, y, r, tmp)
        c.barrier(label="init")
        c.h(x)
        c.barrier(label="prep")
        c.append(f().to_gate(label=f"$U_{{{label}}}$"), [*x, *y])
        c.barrier(label="apply")
        c.measure(y, tmp) # COLLAPSE
        c.h(x)
        c.barrier(label="done")
        c.measure(x, r)
        return c
    def postproc(self, result, option):
        def mod2(x): return x%2
        def mod2zero(x): return x%2 == 0
        r = list(map(lambda x: list(reversed(x.split(" ")[1])), result.get_memory()))
        M = sp.Matrix(r).applyfunc(mod2)
        # iszerofunc is a HACK here
        if M.rank(iszerofunc=mod2zero) < self.i-1:
            return HTML("<font color=red>Not enough executions, better luck next time</font>")
        else:
            ns = M.nullspace(iszerofunc=mod2zero)
            ns = [v.applyfunc(mod2) for v in ns][0]
            return HTML(f"<b>Postrocessing results:</b> s = {ns.tolist()}")


# %% [markdown]
# Alternative approaches to postprocessing.
#
# If you don't mind a package zoo:
# ```python
# import galois
# GF2 = galois.GF(2)
# ns = GF2(result).null_space()
# print(ns[0].tolist())
# ```
#
# SymPy returns an error for this, looks like it's not implemented properly yet:
# ```python
# from sympy.polys.matrices import DomainMatrix
# from sympy.polys.domains import GF
# M_gf2 = DomainMatrix(result, (len(result), len(result[0])), GF(2))
# M_gf2.nullspace()
# ```

# %%
res = CircuitSlicer(Simon(3, 5), nsims=3, common_factors=False, diag=False)

# %%
# circuit check
op = Simon(3, 5).fixed_ibm_example().to_gate()

rows = ""
for i in range(3):
    v = Statevector.from_label(f"{1<<i:08b}")
    rows += rf"""
    <tr><td>${v.draw('latex_source')}$</td><td>$\longrightarrow$</td>
    <td>${v.evolve(op).draw("latex_source")}$</td></tr>"""
display(HTML(f"<table>{rows}</table>"))


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Grover

# %%
class Grover:
    def opt(self, niter):
        oracle = QuantumCircuit(self.dim+1)
        for bits in self.sols:
            zeros = np.arange(self.dim)[bits == 0].tolist()
            if zeros:
                oracle.x(zeros)
            oracle.mcx(list(range(self.dim)) , self.dim)
            if zeros:
                oracle.x(zeros)
        diffuser = QuantumCircuit(self.dim+1)
        diffuser.h(range(self.dim))
        diffuser.x(range(self.dim))
        diffuser.mcx(list(range(self.dim)), self.dim)
        diffuser.x(range(self.dim+1))
        diffuser.h(range(self.dim))
        return (oracle, diffuser)
    def get_options(self):
        # (mis)using partial object to store the number of iterations. opt() ignores its argument
        return [(f"{i} iterations", partial(self.opt, i)) for i in range(1, 2**(self.dim-1))]
    def __init__(self, dim, nsol=1):
        self.dim = dim
        sols = np.random.choice(2**dim, size=nsol, replace=False)
        a = (np.arange(2**dim)[:,None] >> np.arange(dim))&1
        self.sols = a[sols]
        allstates = np.apply_along_axis(lambda x: Statevector.from_label("".join(reversed(x))), 1, a.astype(str))
        u = sum(allstates)/np.sqrt(2**dim)
        self.a0 = sum(allstates[sols])/np.sqrt(nsol)
        self.a1 = sum(np.delete(allstates, sols, axis=0))/np.sqrt(2**dim-nsol)
        self.ux = np.vdot(u, self.a0).real
        self.uy = np.vdot(u, self.a1).real
    def get_circuit(self, opt, label=""):
        x = QuantumRegister(self.dim, "x")
        y = AncillaRegister(1, "y")
        r = ClassicalRegister(self.dim, "r")
        c = QuantumCircuit(x, y, r)
        c.x(y)
        c.h([*x, *y])
        c.barrier(label="init")
        o, d = opt()
        og = o.to_gate(label="$U_f$")
        od = d.to_gate(label="$H Z_{OR} H$")
        for i in range(opt.args[0]):
            c.append(og, [*x, *y])
            c.barrier(label=f"oracle #{i}")
            c.append(od, [*x, *y])
            c.barrier(label=f"diff #{i}")
        c.measure(x, r)
        return c
    def postproc(self, result, options):
        return HTML(f"<b>To guess</b>: {self.sols}")
    def stepproc(self, sv, dm):
        partial_sv = partial_trace(sv, [self.dim]).to_statevector()
        x = partial_sv.inner(self.a0).real
        y = partial_sv.inner(self.a1).real

        fig = Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.text(1.2, 0.1, r"$\vert A_0\rangle$", fontsize=12, ha="center", va="center")
        ax.text(0.15, 1.2, r"$\vert A_1\rangle$", fontsize=12, ha="center", va="center")

        circle = patches.Circle((0, 0), linestyle='--', alpha=0.5, radius=1, fill=False, color="blue")
        ax.add_patch(circle)
        ax.plot((0, self.ux), (0, self.uy), "g")
        ax.text(self.ux*1.1, self.uy*1.1, r"$\vert u\rangle$", fontsize=12, ha="center", va="center")

        ax.plot((0, x), (0, y), "r")
        ax.text(x*1.1, y*1.1, r"$\vert \psi\rangle$", fontsize=12, ha="center", va="center")

        fig.canvas.draw()
        return fig

CircuitSlicer(Grover(6, 1), 1, diag=False, common_factors=False);


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Playground        

# %%
class T:
    def o0(self):
        return [0, 0]
    def get_options(self):
        return [("option 0", self.o0)]
    def get_circuit(self, f, label=""):
        qc = QuantumCircuit(2, 2)
        qc.barrier(label="init")
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier(label="prep")
        qc.measure([0, 1], [0, 1])
        return qc
CircuitSlicer(T());

# %% [markdown]
# Some SymPy experiments
# ```python
# from sympy.physics.quantum import Ket, Dagger, qapply
# from sympy.physics.quantum.qubit import matrix_to_qubit, Qubit
# from sympy.physics.quantum.tensorproduct import TensorProduct
#
# s = sp.Matrix(Statevector.from_label("+01")).applyfunc(partial(sp.nsimplify, constants=[sp.sqrt(2)]))
# m = matrix_to_qubit(s)
# mx = m.expand()
# display(m)
# display(mx)
# def expand_qubits_to_tensor(expr):
#     return expr.subs({
#         Qubit(s): TensorProduct(*[Qubit(bit) for bit in s])
#         for s in [b for b in [bin(i)[2:].zfill(3) for i in range(8)]]})
# t = expand_qubits_to_tensor(m)
# display(t)
# display(TensorProduct.factor(qapply(t)))
#
# #f = sp.gcd(tuple(s))
# #sp.MatMul(f, s/f, evaluate=False)
# ```

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Gates

# %%
import ipywidgets as widgets
x = QuantumRegister(2, "x")
y = AncillaRegister(1, "y")
u_and = QuantumCircuit(x, y, name="AND")
u_and.mcx(x, y)

l = widgets.Output()
with l:
    display(u_and.draw("mpl"))
rows = ""
for v in range(4):
    s = Statevector.from_label(f"{v:03b}")
    rows += fr"""<tr><td>${s.draw('latex_source')}$</td>
    <td>$\longrightarrow$</td>
    <td>${s.evolve(u_and).draw('latex_source')}$</td></tr>"""
display(widgets.HBox([l, widgets.HTMLMath(f"<table>{rows}</table>")], layout=widgets.Layout(align_items='center')))


# %%
def make_or(n):
    x = QuantumRegister(n, "x")
    y = AncillaRegister(1, "y")
    c = QuantumCircuit(x, y, name="OR")
    c.x(x)
    c.mcx(x, y)
    c.x([*x, *y])
    return c

u_or = make_or(2)
l = widgets.Output()
with l:
    display(u_or.draw("mpl"))
rows = ""
for v in range(8):
    s = Statevector.from_label(f"{v:03b}")
    rows += fr"""<tr><td>${s.draw('latex_source')}$</td>
    <td>$\longrightarrow$</td>
    <td>${s.evolve(u_or).draw('latex_source')}$</td></tr>"""
display(widgets.HBox([l, widgets.HTMLMath(f"<table>{rows}</table>")], layout=widgets.Layout(align_items='center')))

# %%
x = QuantumRegister(1, "x")
y = AncillaRegister(1, "0")
u_fanout = QuantumCircuit(x, y, name="fanout")
u_fanout.cx(x, y)

l = widgets.Output()
with l:
    display(u_fanout.draw("mpl"))
rows = ""
for v in range(2):
    s = Statevector.from_label(f"{v:02b}")
    rows += fr"""<tr><td>${s.draw('latex_source')}$</td>
    <td>$\longrightarrow$</td>
    <td>${s.evolve(u_fanout).draw('latex_source')}$</td></tr>"""
display(widgets.HBox([l, widgets.HTMLMath(f"<table>{rows}</table>")], layout=widgets.Layout(align_items='center')))

# %%
from IPython.display import clear_output, HTML, Latex, Math

def make_zf(u):
    c = u.copy_empty_like()
    c.x(c.qubits[-1])
    c.h(c.qubits[-1])
    c.append(u.to_gate(label=f"$U_{{{u.name}}}$"), c.qubits)
    c.h(c.qubits[-1])
    c.x(c.qubits[-1])
    return c

z_or = make_zf(make_or(3))
l = widgets.Output()
with l:
    display(z_or.draw("mpl"))

rows = ""
for v in range(2**(z_or.num_qubits-1)):
    s = Statevector.from_label(f"{v:0{z_or.num_qubits}b}")
    r = s.evolve(z_or)
    dm = partial_trace(r, [z_or.num_qubits-1])
    partial_s = dm.to_statevector()
    rows += fr"""<tr><td>${s.draw('latex_source')}$</td>
    <td>$\longrightarrow$</td><td>${r.draw('latex_source')}$</td>
    <td>:</td><td>${partial_s.draw('latex_source')}$</td></tr>"""
display(widgets.HBox([l, widgets.HTMLMath(f"<table>{rows}</table>")], layout=widgets.Layout(align_items='center')))

# %%
# z_or directly
d = QuantumCircuit(5)
d.x(range(4))
d.mcx(list(range(4)), 4)
d.x(range(5))
rows = ""
l = widgets.Output()
with l:
    display(d.draw("mpl"))
for v in range(2**4):
    s = Statevector.from_label(f"-{v:04b}")
    r = s.evolve(d)
    dm = partial_trace(r, [4])
    partial_s = dm.to_statevector()
    rows += fr"""<tr><td>${s.draw('latex_source')}$</td>
    <td>$\longrightarrow$</td><td>${r.draw('latex_source')}$</td>
    <td>:</td><td>${partial_s.draw('latex_source')}$</td></tr>"""
display(widgets.HBox([l, widgets.HTMLMath(f"<table>{rows}</table>")], layout=widgets.Layout(align_items='center')))

# %% [markdown]
# ## Phase Estimation
