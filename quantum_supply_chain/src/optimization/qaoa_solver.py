import numpy as np
from qiskit import Aer, execute
from qiskit.algorithms import QAOA, VQE
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Estimator
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit.circuit.library import TwoLocal
from qiskit.utils import algorithm_globals
from quantum_supply_chain.src.optimization.tsp_qubo import TSPQUBO  # Import QUBO Model

class QAOASolver:
    def __init__(self, p=3):
        """
        Initializes the QAOA solver with depth p.
        """
        self.p = p
        self.qubo_model = TSPQUBO().get_qubo_model()
        self.backend = Aer.get_backend("aer_simulator")  # Quantum simulator
        self.optimizer = SPSA(maxiter=200)  # Stochastic optimizer
        self.qaoa = None  # QAOA instance
        self.result = None  # Store optimization result

    def solve_with_qaoa(self):
        """
        Solves the TSP QUBO using QAOA.
        """
        estimator = Estimator()
        self.qaoa = QAOA(estimator, reps=self.p, optimizer=self.optimizer)
        qaoa_optimizer = MinimumEigenOptimizer(self.qaoa)
        self.result = qaoa_optimizer.solve(self.qubo_model)
        return self.result

    def solve_with_vqe(self):
        """
        Solves the TSP QUBO using VQE (Variational Quantum Eigensolver).
        """
        var_form = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
        vqe = VQE(var_form, optimizer=self.optimizer)
        vqe_optimizer = MinimumEigenOptimizer(vqe)
        self.result = vqe_optimizer.solve(self.qubo_model)
        return self.result

    def print_results(self):
        """
        Prints the QAOA results.
        """
        if self.result:
            print("Best Quantum Solution Found:")
            print(self.result)

# Example Usage
if __name__ == "__main__":
    solver = QAOASolver(p=3)
    qaoa_result = solver.solve_with_qaoa()
    print("\nQAOA Solution:\n", qaoa_result)
    solver.print_results()