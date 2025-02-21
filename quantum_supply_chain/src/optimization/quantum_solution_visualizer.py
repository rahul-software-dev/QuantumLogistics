import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from qiskit.visualization import plot_histogram
import logging
import time

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class QuantumSolutionVisualizer:
    """
    Advanced Visualization for Quantum Optimization Solutions.
    
    Features:
    - Route visualization (Classical vs Quantum)
    - Animated energy landscape plotting (Quantum Optimization)
    - Quantum state probability visualization (3D & 2D)
    - Heatmap for QUBO formulation
    - Performance comparison (Cost, Time, Efficiency)
    """

    def __init__(self, graph: nx.Graph, classical_solution=None, quantum_solution=None, qaoa_probabilities=None, energy_levels=None):
        """
        Initializes the visualizer.
        
        :param graph: Supply Chain or Optimization Graph
        :param classical_solution: Best classical algorithm solution
        :param quantum_solution: Best QAOA/VQE-based solution
        :param qaoa_probabilities: Quantum state probabilities (from QAOA)
        :param energy_levels: QAOA/VQE energy levels over iterations
        """
        self.graph = graph
        self.classical_solution = classical_solution
        self.quantum_solution = quantum_solution
        self.qaoa_probabilities = qaoa_probabilities
        self.energy_levels = energy_levels

    ### === 1. Route Visualization === ###
    
    def visualize_routes(self):
        """Plots Classical vs Quantum optimized routes on an interactive graph."""
        fig = go.Figure()

        # Node Positions
        pos = nx.spring_layout(self.graph)

        # Add Nodes
        for node, coords in pos.items():
            fig.add_trace(go.Scatter(
                x=[coords[0]], y=[coords[1]], mode="markers+text",
                marker=dict(size=12, color="black"), name=f"Node {node}",
                text=f"Node {node}", textposition="bottom center"
            ))

        # Classical Route
        if self.classical_solution:
            classical_edges = [(self.classical_solution[i], self.classical_solution[i + 1]) for i in range(len(self.classical_solution) - 1)]
            for edge in classical_edges:
                fig.add_trace(go.Scatter(
                    x=[pos[edge[0]][0], pos[edge[1]][0]],
                    y=[pos[edge[0]][1], pos[edge[1]][1]],
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="Classical Path"
                ))

        # Quantum Route
        if self.quantum_solution:
            quantum_edges = [(self.quantum_solution[i], self.quantum_solution[i + 1]) for i in range(len(self.quantum_solution) - 1)]
            for edge in quantum_edges:
                fig.add_trace(go.Scatter(
                    x=[pos[edge[0]][0], pos[edge[1]][0]],
                    y=[pos[edge[0]][1], pos[edge[1]][1]],
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name="Quantum Path"
                ))

        fig.update_layout(title="Classical vs Quantum Optimized Routes", showlegend=True)
        fig.show()

    ### === 2. Quantum Energy Landscape (Animated) === ###
    
    def animate_energy_landscape(self):
        """Creates an animated plot showing energy optimization convergence."""
        if not self.energy_levels:
            logging.warning("No energy data available for visualization.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.set_title("Quantum Energy Landscape Convergence")

        x_data = []
        y_data = []
        for i, energy in enumerate(self.energy_levels):
            x_data.append(i)
            y_data.append(energy)
            ax.clear()
            ax.plot(x_data, y_data, marker="o", color="purple")
            plt.pause(0.1)

        plt.show()

    ### === 3. Quantum State Probability (3D & 2D) === ###
    
    def plot_qaoa_probabilities(self):
        """Plots a 3D quantum state probability visualization."""
        if not self.qaoa_probabilities:
            logging.warning("No QAOA probabilities available for visualization.")
            return

        states = list(self.qaoa_probabilities.keys())
        probabilities = list(self.qaoa_probabilities.values())
        x_vals = np.arange(len(states))

        fig = go.Figure(data=[go.Bar(
            x=states, y=probabilities, text=probabilities, textposition="auto",
            marker=dict(color=probabilities, colorscale="Viridis")
        )])

        fig.update_layout(title="Quantum State Probabilities (QAOA)", xaxis_title="States", yaxis_title="Probability")
        fig.show()

    ### === 4. Heatmap for QUBO Problem === ###
    
    def plot_qubo_heatmap(self, qubo_matrix):
        """Plots a heatmap representing the QUBO formulation."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(qubo_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
        plt.title("QUBO Matrix Heatmap")
        plt.show()

    ### === 5. Performance Comparison === ###

    def compare_solutions(self):
        """Compares the cost and time efficiency of classical vs quantum solutions."""
        classical_time_start = time.time()
        classical_cost = self._calculate_route_cost(self.classical_solution)
        classical_time = time.time() - classical_time_start

        quantum_time_start = time.time()
        quantum_cost = self._calculate_route_cost(self.quantum_solution)
        quantum_time = time.time() - quantum_time_start

        logging.info(f"🔵 Classical Optimization Cost: {classical_cost} (Time: {classical_time:.4f}s)")
        logging.info(f"🔴 Quantum Optimization Cost: {quantum_cost} (Time: {quantum_time:.4f}s)")

        print(f"\n🔵 Classical Route Cost: {classical_cost}, Time Taken: {classical_time:.4f}s")
        print(f"🔴 Quantum Route Cost: {quantum_cost}, Time Taken: {quantum_time:.4f}s")

        improvement = ((classical_cost - quantum_cost) / classical_cost) * 100
        print(f"✅ Quantum Improvement: {round(improvement, 2)}%")

    def _calculate_route_cost(self, route):
        """Computes the cost of a given route."""
        return sum(self.graph[route[i]][route[i + 1]]['weight'] for i in range(len(route) - 1)) if route else float('inf')


if __name__ == "__main__":
    # Sample Graph
    G = nx.Graph()
    G.add_edge(0, 1, weight=12)
    G.add_edge(1, 2, weight=15)
    G.add_edge(2, 3, weight=10)
    G.add_edge(3, 0, weight=20)
    G.add_edge(1, 3, weight=18)

    # Example Solutions
    classical_route = [0, 1, 2, 3]
    quantum_route = [0, 3, 2, 1]
    qaoa_probs = {"000": 0.3, "011": 0.4, "101": 0.2, "110": 0.1}
    energy_levels = np.linspace(-10, -1, 10)

    visualizer = QuantumSolutionVisualizer(G, classical_solution=classical_route, quantum_solution=quantum_route, qaoa_probabilities=qaoa_probs, energy_levels=energy_levels)

    visualizer.visualize_routes()
    visualizer.animate_energy_landscape()
    visualizer.plot_qaoa_probabilities()
    visualizer.compare_solutions()