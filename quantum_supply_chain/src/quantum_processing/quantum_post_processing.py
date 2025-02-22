import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import List, Dict, Tuple

class QuantumPostProcessing:
    """
    Analyzes, validates, and visualizes quantum optimization results for supply chain logistics.
    """

    def __init__(self, quantum_results: Dict[str, float], graph: nx.Graph, classical_solution: List[int] = None):
        """
        :param quantum_results: Dictionary mapping routes to probabilities (quantum result output).
        :param graph: NetworkX Graph (Directed or Undirected) representing the supply chain.
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
        Handles cases where edges are missing or graphs are directed.
        :param route: List of nodes representing a route.
        :return: Total route cost.
        """
        total_cost = 0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.graph.has_edge(u, v):
                total_cost += self.graph[u][v].get("weight", 1)  # Default weight to 1 if missing
            else:
                print(f"⚠ Missing Edge: ({u} → {v}), assigning high penalty!")
                total_cost += 999  # Assign a high penalty for missing edges
        return total_cost

    def compare_with_classical(self):
        """
        Compares the quantum solution with the classical approach.
        Provides a multi-metric comparison including efficiency, stability, and cost reduction.
        """
        if self.classical_solution is None:
            print("⚠ No classical solution provided for comparison.")
            return

        classical_cost = self.compute_route_cost(self.classical_solution)

        print(f"\n📊 **Quantum vs. Classical Comparison**")
        print(f"🔹 Quantum Route: {self.optimal_route} (Cost: {self.optimal_cost})")
        print(f"🔹 Classical Route: {self.classical_solution} (Cost: {classical_cost})")

        improvement = ((classical_cost - self.optimal_cost) / classical_cost) * 100
        stability = self.compute_stability()

        print(f"🚀 **Quantum Improvement:** {improvement:.2f}%")
        print(f"⚖ **Quantum Solution Stability:** {stability:.2f}")

    def compute_stability(self) -> float:
        """
        Computes the stability of the quantum results using variance and entropy.
        :return: Stability score (lower variance & higher entropy = more stable).
        """
        probabilities = np.array(list(self.quantum_results.values()))
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Avoid log(0)
        variance = np.var(probabilities)
        skewness = stats.skew(probabilities)

        print(f"📈 Quantum Probability Distribution: Entropy = {entropy:.4f}, Variance = {variance:.4f}, Skewness = {skewness:.4f}")

        return entropy / (variance + 1e-6)  # Normalize with small epsilon to avoid division by zero

    def visualize_route(self):
        """
        Visualizes the quantum-optimized supply chain route.
        Uses an improved layout and color-coded node distinction.
        """
        pos = nx.spring_layout(self.graph, seed=42)  # Stable layout
        plt.figure(figsize=(10, 7))

        # Draw the full graph
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=600, font_size=12)

        # Highlight the optimal quantum route
        edges = [(self.optimal_route[i], self.optimal_route[i + 1]) for i in range(len(self.optimal_route) - 1)]
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color="red", width=3)

        plt.title("🔍 Quantum-Optimized Supply Chain Route")
        plt.show()

    def plot_probability_distribution(self):
        """
        Plots the probability distribution of different quantum solutions.
        Enhances readability with interactive Matplotlib.
        """
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5))

        routes = list(self.quantum_results.keys())
        probabilities = list(self.quantum_results.values())

        sns.barplot(x=routes, y=probabilities, palette="coolwarm")
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Route", fontsize=12)
        plt.ylabel("Probability", fontsize=12)
        plt.title("🧬 Quantum Route Probability Distribution", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)

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

    # Create a sample supply chain graph (supports both directed and undirected)
    G = nx.DiGraph()  # Change to nx.Graph() for undirected networks
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