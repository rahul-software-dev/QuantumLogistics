from qiskit_optimization.applications import Tsp
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA
import numpy as np

# 3-city distance matrix
w = np.array([[0, 1, 2],
              [1, 0, 1],
              [2, 1, 0]])
tsp = Tsp(w)
qp = tsp.to_quadratic_program()
print("Num binary vars:", qp.get_num_binary_vars())

backend = Aer.get_backend('aer_simulator_statevector')
q_instance = QuantumInstance(backend)
qaoa = QAOA(optimizer=COBYLA(), reps=1, quantum_instance=q_instance)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)
print("QAOA result:", result.x, result.fval)
