import numpy as np
import networkx as nx
import requests
import time
import random
from qiskit import Aer, execute
from qiskit.optimization.applications.ising import tsp
from src.optimization.qaoa_solver import QAOASolver
from src.optimization.classical_solver import ClassicalSolver
from src.security.quantum_security import QuantumSecureBlockchain

class RealTimeQuantumOptimization:
    """
    Uses real-time data and quantum computing to optimize supply chain logistics dynamically.
    """

    def __init__(self, supply_chain_graph: nx.Graph):
        """
        :param supply_chain_graph: NetworkX Graph representing supply chain nodes and edges.
        """
        self.graph = supply_chain_graph
        self.qaoa_solver = QAOASolver()
        self.classical_solver = ClassicalSolver()
        self.blockchain = QuantumSecureBlockchain()
        self.current_optimal_route = None
        self.current_cost = float('inf')

    def fetch_real_time_data(self) -> dict:
        """
        Simulates fetching real-time logistics data (traffic, demand, weather) from APIs.
        :return: Dictionary containing real-time factors affecting logistics.
        """
        # Simulate traffic, weather, and demand impact on supply chain nodes
        real_time_data = {
            "traffic": {edge: random.uniform(0.8, 1.2) for edge in self.graph.edges},
            "weather": {node: random.uniform(0.9, 1.1) for node in self.graph.nodes},
            "demand_fluctuation": {node: random.uniform(0.8, 1.2) for node in self.graph.nodes}
        }
        return real_time_data

    def adjust_graph_weights(self, real_time_data: dict):
        """
        Adjusts graph weights dynamically based on real-time conditions.
        :param real_time_data: Live data affecting logistics (traffic, weather, demand).
        """
        for edge in self.graph.edges:
            traffic_factor = real_time_data["traffic"][edge]
            self.graph[edge[0]][edge[1]]["weight"] *= traffic_factor  # Adjust weight based on traffic
        
        for node in self.graph.nodes:
            weather_factor = real_time_data["weather"][node]
            demand_factor = real_time_data["demand_fluctuation"][node]
            # Adjust node-based conditions (if applicable in your model)

    def optimize_logistics(self, method="quantum") -> tuple:
        """
        Optimizes the supply chain routing in real-time using quantum or classical methods.
        :param method: "quantum" (QAOA) or "classical" (Dijkstra).
        :return: Optimal route and its cost.
        """
        if method == "quantum":
            optimal_route, cost = self.qaoa_solver.solve_tsp(self.graph)
        else:
            optimal_route, cost = self.classical_solver.solve_tsp(self.graph)
        
        # Update if new route is better
        if cost < self.current_cost:
            self.current_optimal_route = optimal_route
            self.current_cost = cost
            print(f"🔹 **Updated Optimal Route:** {optimal_route} (Cost: {cost})")

        return optimal_route, cost

    def secure_decisions_with_blockchain(self):
        """
        Stores optimized decisions on a blockchain to prevent tampering.
        """
        decision_data = {
            "timestamp": time.time(),
            "route": self.current_optimal_route,
            "cost": self.current_cost
        }
        self.blockchain.store_decision(decision_data)
        print("✅ Decision securely stored on blockchain.")

    def run_real_time_optimization(self, update_interval=30):
        """
        Continuously updates the supply chain optimization based on live data.
        :param update_interval: Time interval (in seconds) for fetching new data.
        """
        while True:
            print("\n🔄 Fetching real-time supply chain data...")
            real_time_data = self.fetch_real_time_data()
            self.adjust_graph_weights(real_time_data)

            print("🚀 Optimizing logistics with quantum computing...")
            self.optimize_logistics(method="quantum")

            print("🔐 Securing decision with blockchain...")
            self.secure_decisions_with_blockchain()

            time.sleep(update_interval)  # Wait before next update


# Example Usage
if __name__ == "__main__":
    # Create a sample supply chain graph
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 10), (0, 2, 15), (1, 3, 12), (2, 3, 8), (3, 4, 7)])

    # Initialize the real-time quantum optimizer
    real_time_optimizer = RealTimeQuantumOptimization(G)

    # Run continuous real-time optimization
    real_time_optimizer.run_real_time_optimization(update_interval=60)