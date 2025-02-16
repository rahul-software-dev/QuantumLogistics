import networkx as nx
import numpy as np
import time
from src.optimization.qaoa_solver import QAOASolver
from src.optimization.classical_solver import ClassicalSolver

class HybridQuantumClassicalSolver:
    """
    Hybrid solver that dynamically selects between quantum (QAOA) and classical (Dijkstra/Genetic) approaches
    based on problem complexity and real-time constraints.
    """

    def __init__(self, graph: nx.Graph, quantum_threshold=10, use_qaoa=True):
        """
        :param graph: The supply chain network as a NetworkX graph.
        :param quantum_threshold: Number of nodes above which quantum computing is used.
        :param use_qaoa: Boolean to enable/disable quantum optimization.
        """
        self.graph = graph
        self.qaoa_solver = QAOASolver()
        self.classical_solver = ClassicalSolver()
        self.quantum_threshold = quantum_threshold
        self.use_qaoa = use_qaoa

    def evaluate_problem_complexity(self) -> str:
        """
        Determines whether to use a quantum or classical approach based on problem complexity.
        :return: 'quantum' or 'classical' decision
        """
        num_nodes = len(self.graph.nodes)
        num_edges = len(self.graph.edges)
        density = num_edges / (num_nodes * (num_nodes - 1) / 2)  # Graph density

        if num_nodes >= self.quantum_threshold and self.use_qaoa:
            return "quantum"
        elif density > 0.6:  # Highly connected graph → Classical solvers are efficient
            return "classical"
        return "hybrid"

    def solve_tsp(self, method=None) -> tuple:
        """
        Solves the Traveling Salesman Problem (TSP) using either quantum, classical, or hybrid approach.
        :param method: 'quantum', 'classical', or 'hybrid'. If None, it will auto-select.
        :return: Optimal route and its cost.
        """
        if method is None:
            method = self.evaluate_problem_complexity()

        print(f"🔍 Selected Optimization Method: {method.upper()}")

        start_time = time.time()

        if method == "quantum":
            optimal_route, cost = self.qaoa_solver.solve_tsp(self.graph)
        elif method == "classical":
            optimal_route, cost = self.classical_solver.solve_tsp(self.graph)
        else:  # Hybrid Approach
            # First, run a fast classical heuristic
            heuristic_route, heuristic_cost = self.classical_solver.solve_tsp(self.graph, heuristic=True)

            # Run QAOA only if classical heuristic is inefficient
            if heuristic_cost > 1.2 * heuristic_cost:  # If classical is significantly worse
                print("⚡ Switching to QAOA for deeper optimization...")
                optimal_route, cost = self.qaoa_solver.solve_tsp(self.graph)
            else:
                optimal_route, cost = heuristic_route, heuristic_cost

        end_time = time.time()
        print(f"✅ Optimization Completed in {end_time - start_time:.4f} seconds.")
        return optimal_route, cost


# Example Usage
if __name__ == "__main__":
    # Sample Supply Chain Graph
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 10), (0, 2, 15), (1, 3, 12), (2, 3, 8), (3, 4, 7)])

    # Initialize the Hybrid Solver
    hybrid_solver = HybridQuantumClassicalSolver(G, quantum_threshold=5)

    # Solve TSP using hybrid approach
    optimal_route, cost = hybrid_solver.solve_tsp()
    print(f"🚀 Optimal Route: {optimal_route} (Cost: {cost})")