import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

class QuantumPostProcessing:
    """
    Analyzes, validates, and visualizes quantum optimization results for supply chain logistics.
    """

    def __init__(self, quantum_results: Dict[str, float], graph: nx.Graph, classical_solution: List[int] = None):
        """
        :param quantum_results: Dictionary mapping routes to probabilities (quantum result output).
        :param graph: NetworkX Graph representing the supply chain.
        :param classical_solution: Classical solver result for comparison (optional).
        """
        self.quantum_results = quantum_results
        self.graph = graph
        self.classical_solution = classical_solution
        self.optimal_route = None
        self.optimal_cost = None

    def extract_optimal_route(self) -> Tuple[List[int], float]:
        """
        Extracts the most probable route from quantum results.
        :return: Optimal route (node sequence) and its cost.
        """
        sorted_results = sorted(self.quantum_results.items(), key=lambda x: -x[1])  # Sort by probability
        self.optimal_route = list(map(int, sorted_results[0][0].split("-")))
        
        # Compute cost based on the extracted route
        self.optimal_cost = self.compute_route_cost(self.optimal_route)
        print(f"✔ Optimal Quantum Route: {self.optimal_route} (Cost: {self.optimal_cost})")
        
        return self.optimal_route, self.optimal_cost

    def compute_route_cost(self, route: List[int]) -> float:
        """
        Computes the cost of a given route using edge weights from the graph.
        :param route: List of nodes representing a route.
        :return: Total route cost.
        """
        cost = sum(self.graph[route[i]][route[i + 1]]["weight"] for i in range(len(route) - 1))
        return cost

    def compare_with_classical(self):
        """
        Compares the quantum solution with the classical approach.
        """
        if self.classical_solution is None:
            print("⚠ No classical solution provided for comparison.")
            return

        classical_cost = self.compute_route_cost(self.classical_solution)

        print(f"\n📊 **Quantum vs. Classical Comparison**")
        print(f"🔹 Quantum Route: {self.optimal_route} (Cost: {self.optimal_cost})")
        print(f"🔹 Classical Route: {self.classical_solution} (Cost: {classical_cost})")

        improvement = ((classical_cost - self.optimal_cost) / classical_cost) * 100
        print(f"🚀 **Quantum Improvement:** {improvement:.2f}%")

    def visualize_route(self):
        """
        Visualizes the quantum-optimized supply chain route.
        """
        pos = nx.spring_layout(self.graph)  # Position nodes for visualization
        plt.figure(figsize=(10, 7))

        # Draw the full graph
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500)

        # Highlight the optimal quantum route
        edges = [(self.optimal_route[i], self.optimal_route[i + 1]) for i in range(len(self.optimal_route) - 1)]
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color="red", width=2)

        plt.title("🔍 Quantum-Optimized Supply Chain Route")
        plt.show()

    def plot_probability_distribution(self):
        """
        Plots the probability distribution of different quantum solutions.
        """
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5))

        routes = list(self.quantum_results.keys())
        probabilities = list(self.quantum_results.values())

        sns.barplot(x=routes, y=probabilities, palette="coolwarm")
        plt.xticks(rotation=90)
        plt.xlabel("Route")
        plt.ylabel("Probability")
        plt.title("🧬 Quantum Route Probability Distribution")
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Sample quantum results: {"route_string": probability}
    quantum_results = {
        "0-2-3-1-4": 0.45,
        "0-3-2-1-4": 0.30,
        "0-1-3-2-4": 0.15,
        "0-2-1-3-4": 0.10
    }

    # Create a sample supply chain graph
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 10), (0, 2, 15), (1, 3, 12), (2, 3, 8), (3, 4, 7)])

    # Sample classical solution
    classical_solution = [0, 2, 3, 1, 4]

    # Initialize Post-Processing
    post_processor = QuantumPostProcessing(quantum_results, G, classical_solution)

    # Extract the best quantum route
    post_processor.extract_optimal_route()

    # Compare with classical solution
    post_processor.compare_with_classical()

    # Visualize the optimal quantum route
    post_processor.visualize_route()

    # Plot probability distribution
    post_processor.plot_probability_distribution()