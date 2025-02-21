import networkx as nx
import numpy as np
from itertools import permutations
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel, ExactSolver
from scipy.optimize import basinhopping
import time

class AnnealingSolver:
    """
    Quantum Annealing-based Solver for TSP/VRP using D-Wave.
    Includes:
    - Classical Simulated Annealing for comparison.
    - Quantum Annealing via QUBO formulation.
    - Hybrid approach (Classical + Quantum refinement).
    """

    def __init__(self, graph: nx.Graph, num_nodes: int, chain_strength=2.5, num_reads=1500, quantum_backend="D-Wave"):
        """
        Initializes the solver with a supply chain graph.
        :param graph: NetworkX Graph representing the supply chain network.
        :param num_nodes: Number of nodes in the graph.
        :param chain_strength: Quantum annealing chain strength (adjustable parameter).
        :param num_reads: Number of times the quantum annealer runs.
        :param quantum_backend: "D-Wave" or "Simulator".
        """
        self.graph = graph
        self.num_nodes = num_nodes
        self.chain_strength = chain_strength
        self.num_reads = num_reads
        self.quantum_backend = quantum_backend

    def preprocess_graph(self):
        """
        Preprocesses the graph by ensuring it is fully connected.
        Missing edges are replaced with a high penalty weight.
        """
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and not self.graph.has_edge(i, j):
                    self.graph.add_edge(i, j, weight=9999)  # High cost for missing edges

    def classical_simulated_annealing(self):
        """
        Solves TSP using Simulated Annealing as a classical benchmark.
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
        result = basinhopping(energy_function, initial_route, minimizer_kwargs=minimizer_kwargs, niter=150)

        return result.x, result.fun

    def qubo_formulation(self):
        """
        Converts TSP into QUBO formulation for Quantum Annealing.
        Returns a BinaryQuadraticModel (BQM) for D-Wave.
        """
        qubo = {}
        for i, j in permutations(range(self.num_nodes), 2):
            qubo[(i, j)] = self.graph[i][j]['weight']

        # Convert QUBO dictionary into a BinaryQuadraticModel
        return BinaryQuadraticModel.from_qubo(qubo)

    def quantum_annealing_solver(self):
        """
        Uses D-Wave Quantum Annealer or a Classical QUBO Solver if no quantum hardware is available.
        """

        qubo_model = self.qubo_formulation()

        if self.quantum_backend == "D-Wave":
            try:
                sampler = EmbeddingComposite(DWaveSampler())
                response = sampler.sample(qubo_model, chain_strength=self.chain_strength, num_reads=self.num_reads)

                # Extract best solution
                best_solution = response.first.sample
                best_energy = response.first.energy
                return best_solution, best_energy
            except Exception as e:
                print(f"Quantum Annealing Error: {e}")
                return None, None

        elif self.quantum_backend == "Simulator":
            print("Using Exact Classical Solver for QUBO...")
            sampler = ExactSolver()
            response = sampler.sample(qubo_model)

            best_solution = response.first.sample
            best_energy = response.first.energy
            return best_solution, best_energy

        else:
            raise ValueError("Invalid quantum backend. Choose 'D-Wave' or 'Simulator'.")

    def hybrid_solver(self):
        """
        Runs a Hybrid Optimization: First Classical, Then Quantum for Refinement.
        """
        classical_solution, classical_cost = self.classical_simulated_annealing()
        quantum_solution, quantum_cost = self.quantum_annealing_solver()

        if quantum_solution is None:
            print("Quantum solver failed, using classical-only results.")
            return {"solution": classical_solution, "cost": classical_cost}

        hybrid_solution = quantum_solution  # Placeholder: Future work on Hybrid refinement

        return {
            "classical_solution": classical_solution,
            "classical_cost": classical_cost,
            "quantum_solution": quantum_solution,
            "quantum_cost": quantum_cost,
            "hybrid_solution": hybrid_solution
        }

if __name__ == "__main__":
    # Sample Supply Chain Graph with Distances
    G = nx.Graph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(1, 2, weight=20)
    G.add_edge(2, 3, weight=15)
    G.add_edge(3, 0, weight=25)

    start_time = time.time()
    solver = AnnealingSolver(graph=G, num_nodes=4, quantum_backend="D-Wave")
    solver.preprocess_graph()
    results = solver.hybrid_solver()
    end_time = time.time()

    print("\n🔍 **Results:**")
    print(f"🔹 Classical Simulated Annealing Route: {results['classical_solution']} | Cost: {results['classical_cost']}")
    print(f"🔹 Quantum Annealing Route: {results['quantum_solution']} | Cost: {results['quantum_cost']}")
    print(f"🔹 Hybrid Optimized Solution: {results['hybrid_solution']}")
    print(f"⏱ Execution Time: {round(end_time - start_time, 2)} sec")