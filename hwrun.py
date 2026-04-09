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
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(token="xx")

# %%
from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2
from matplotlib import pyplot as plt
# from qiskit_ibm_runtime.fake_provider import FakeBelemV2

# %%
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.draw("mpl")

# %%
observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
service = QiskitRuntimeService()

backend = service.least_busy(simulator=False, operational=True)

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

isa_circuit.draw("mpl", idle_wires=False)

# %%
estimator = Estimator(mode=backend)
estimator.options.resilience_level = 1
estimator.options.default_shots = 5000

mapped_observables = [observable.apply_layout(isa_circuit.layout) for observable in observables]
job = estimator.run([(isa_circuit, mapped_observables)])

print(f">>> Job ID: {job.job_id()}")

# %%
job_result = job.result()
pub_result = job.result()[0]

values = pub_result.data.evs
errors = pub_result.data.stds

plt.plot(observables_labels, values, "-o")
plt.xlabel("Observables")
plt.ylabel("Values")
plt.show()

# %%
a = QuantumRegister(1, "a")
b = QuantumRegister(1, "b")
r = ClassicalRegister(2, "r")
qc = QuantumCircuit(a, b, r)
qc.h(0)
qc.cx(0, 1)
qc.measure(a, r[0])
qc.measure(b, r[1])

# %%
transpiled_qc = transpile(qc, backend=backend, optimization_level=1)
sampler = SamplerV2(mode=backend)
job = sampler.run([transpiled_qc])
result = job.result()

pub_result = result[0]

# %%
pub_result.data.r.get_counts()
