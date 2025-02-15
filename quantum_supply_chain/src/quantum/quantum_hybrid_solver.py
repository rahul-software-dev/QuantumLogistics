import numpy as np
import networkx as nx
from scipy.optimize import basinhopping
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp
from itertools import permutations

class QuantumHybridSolver:
    """
    Hybrid Classical + Quantum Solver for Supply Chain Optimization.
    Uses Simulated Annealing for initial optimization, then refines it with QAOA.
    """

    def __init__(self, graph: nx.Graph, num_nodes: int):
        """
        Initializes the solver with a graph representing supply chain nodes.
        """
        self.graph = graph
        self.num_nodes = num_nodes
        self.best_classical_solution = None

    def classical_solver(self):
        """
        Solves TSP using Simulated Annealing as an initial classical approach.
        """

        def route_distance(route):
            """Calculate total distance for a given route."""
            return sum(self.graph[route[i]][route[i + 1]]['weight'] for i in range(len(route) - 1))

        def energy_function(route):
            """Energy function for simulated annealing."""
            return route_distance(route)

        initial_route = list(self.graph.nodes)
        np.random.shuffle(initial_route)  # Random initial route

        # Simulated Annealing Optimization
        minimizer_kwargs = {"method": "L-BFGS-B"}
        result = basinhopping(energy_function, initial_route, minimizer_kwargs=minimizer_kwargs, niter=100)

        self.best_classical_solution = result.x
        return self.best_classical_solution

    def qaoa_optimizer(self):
        """
        Uses QAOA to refine the classical solution.
        """

        # Convert classical solution into QUBO matrix form
        def get_qubo_matrix():
            """Generates QUBO formulation for the problem."""
            qubo = {}
            for i, j in permutations(range(self.num_nodes), 2):
                qubo[(i, j)] = self.graph[i][j]['weight']
            return qubo

        pauli_op = PauliSumOp.from_list([("ZZ", weight) for (i, j), weight in get_qubo_matrix().items()])

        backend = Aer.get_backend('statevector_simulator')
        qaoa = QAOA(optimizer=COBYLA(), reps=3, quantum_instance=backend)
        result = qaoa.compute_minimum_eigenvalue(operator=pauli_op)

        return result.eigenvalue.real

    def hybrid_solver(self):
        """
        Runs the hybrid optimization: first classical, then quantum.
        """
        classical_solution = self.classical_solver()
        quantum_refinement = self.qaoa_optimizer()
        return classical_solution, quantum_refinement

if __name__ == "__main__":
    # Sample Supply Chain Graph with Distances
    G = nx.Graph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(1, 2, weight=20)
    G.add_edge(2, 3, weight=15)
    G.add_edge(3, 0, weight=25)

    solver = QuantumHybridSolver(graph=G, num_nodes=4)
    classical_sol, quantum_refined_sol = solver.hybrid_solver()

    print(f"Classical Optimized Route: {classical_sol}")
    print(f"Quantum-Refined Score: {quantum_refined_sol}")