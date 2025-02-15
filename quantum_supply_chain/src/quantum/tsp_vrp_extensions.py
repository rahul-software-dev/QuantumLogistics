import numpy as np
import networkx as nx
from itertools import permutations
from scipy.optimize import basinhopping
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp
import random

class TSP_VRP_Solver:
    """
    Solver for both Traveling Salesman Problem (TSP) and Vehicle Routing Problem (VRP).
    Uses classical heuristics and quantum computing approaches.
    """

    def __init__(self, graph: nx.Graph, num_vehicles: int):
        """
        Initializes the solver with a supply chain graph and vehicle count.
        """
        self.graph = graph
        self.num_vehicles = num_vehicles
        self.best_routes = None

    def clarke_wright_savings(self):
        """
        Classical Clarke-Wright Savings algorithm for VRP.
        This partitions the supply chain nodes for multiple vehicle routes.
        """
        nodes = list(self.graph.nodes)
        depot = nodes[0]  # Assuming first node as the depot
        savings = {}

        for i, j in permutations(nodes[1:], 2):
            direct_cost = self.graph[depot][i]["weight"] + self.graph[depot][j]["weight"]
            combined_cost = self.graph[i][j]["weight"]
            savings[(i, j)] = direct_cost - combined_cost

        sorted_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)
        vehicle_routes = {v: [depot] for v in range(self.num_vehicles)}

        assigned_nodes = set()
        for (i, j), _ in sorted_savings:
            if i not in assigned_nodes and j not in assigned_nodes:
                for v in range(self.num_vehicles):
                    if len(vehicle_routes[v]) < len(nodes) // self.num_vehicles + 1:
                        vehicle_routes[v].extend([i, j])
                        assigned_nodes.update([i, j])
                        break

        for v in range(self.num_vehicles):
            vehicle_routes[v].append(depot)

        self.best_routes = vehicle_routes
        return vehicle_routes

    def tsp_simulated_annealing(self, route):
        """
        Optimizes a single vehicle’s route using Simulated Annealing.
        """

        def route_distance(route):
            return sum(self.graph[route[i]][route[i + 1]]['weight'] for i in range(len(route) - 1))

        def energy_function(route):
            return route_distance(route)

        initial_route = route[:]
        np.random.shuffle(initial_route)

        minimizer_kwargs = {"method": "L-BFGS-B"}
        result = basinhopping(energy_function, initial_route, minimizer_kwargs=minimizer_kwargs, niter=100)

        return result.x

    def qaoa_vrp_optimizer(self):
        """
        Uses QAOA to optimize vehicle assignments and routes.
        """

        def get_qubo_matrix():
            qubo = {}
            for i, j in permutations(range(len(self.best_routes)), 2):
                qubo[(i, j)] = self.graph[i][j]["weight"]
            return qubo

        pauli_op = PauliSumOp.from_list([("ZZ", weight) for (i, j), weight in get_qubo_matrix().items()])

        backend = Aer.get_backend("statevector_simulator")
        qaoa = QAOA(optimizer=COBYLA(), reps=3, quantum_instance=backend)
        result = qaoa.compute_minimum_eigenvalue(operator=pauli_op)

        return result.eigenvalue.real

    def hybrid_vrp_solver(self):
        """
        Runs the hybrid optimization: first Clarke-Wright, then simulated annealing, then QAOA.
        """
        initial_routes = self.clarke_wright_savings()
        optimized_routes = {v: self.tsp_simulated_annealing(initial_routes[v]) for v in initial_routes}
        quantum_refinement = self.qaoa_vrp_optimizer()

        return optimized_routes, quantum_refinement


if __name__ == "__main__":
    # Sample Supply Chain Graph with Distances
    G = nx.Graph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(1, 2, weight=20)
    G.add_edge(2, 3, weight=15)
    G.add_edge(3, 4, weight=30)
    G.add_edge(4, 5, weight=25)
    G.add_edge(5, 0, weight=35)

    solver = TSP_VRP_Solver(graph=G, num_vehicles=2)
    optimized_routes, quantum_refined_score = solver.hybrid_vrp_solver()

    print(f"Optimized Vehicle Routes: {optimized_routes}")
    print(f"Quantum-Refined Cost Reduction Score: {quantum_refined_score}")