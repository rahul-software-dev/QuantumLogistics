import networkx as nx
import numpy as np
from itertools import permutations
from dimod import BinaryQuadraticModel

class QUBOFormulation:
    """
    Generalized QUBO Formulation for multiple combinatorial optimization problems.
    Supports TSP, VRP, and Knapsack by converting them into QUBO models.
    """

    def __init__(self, graph: nx.Graph, problem_type="TSP", penalty_factor=5.0):
        """
        Initializes the QUBO model for a given problem.
        :param graph: NetworkX Graph (used for TSP & VRP).
        :param problem_type: Type of problem ("TSP", "VRP", "Knapsack").
        :param penalty_factor: Dynamic penalty factor to enforce constraints.
        """
        self.graph = graph
        self.problem_type = problem_type
        self.penalty_factor = penalty_factor
        self.bqm = BinaryQuadraticModel('BINARY')

    def preprocess_graph(self):
        """
        Ensures graph connectivity and assigns missing edge weights.
        """
        for i in self.graph.nodes:
            for j in self.graph.nodes:
                if i != j and not self.graph.has_edge(i, j):
                    self.graph.add_edge(i, j, weight=9999)  # Assign high cost to missing edges

    def adaptive_penalty_factor(self):
        """
        Dynamically adjusts penalty factor based on graph complexity.
        """
        avg_weight = np.mean([self.graph[i][j]['weight'] for i, j in self.graph.edges])
        return max(self.penalty_factor, avg_weight * 1.5)

    def tsp_qubo(self):
        """
        Converts the Traveling Salesman Problem (TSP) into a QUBO model.
        """
        num_nodes = len(self.graph.nodes)
        x = {(i, t): f"x_{i}_{t}" for i in range(num_nodes) for t in range(num_nodes)}
        penalty = self.adaptive_penalty_factor()

        # Objective: Minimize total travel cost
        for i, j in permutations(range(num_nodes), 2):
            weight = self.graph[i][j]['weight']
            for t in range(num_nodes - 1):
                self.bqm.add_interaction(x[(i, t)], x[(j, t + 1)], weight)

        # Constraint 1: Each city appears exactly once in the route
        for i in range(num_nodes):
            constraint = sum(self.bqm.add_variable(x[(i, t)], -penalty) for t in range(num_nodes))
            self.bqm.add_offset(penalty * (constraint ** 2))

        # Constraint 2: Each position in the tour is occupied
        for t in range(num_nodes):
            constraint = sum(self.bqm.add_variable(x[(i, t)], -penalty) for i in range(num_nodes))
            self.bqm.add_offset(penalty * (constraint ** 2))

        return self.bqm

    def vrp_qubo(self, vehicle_count=2):
        """
        Converts the Vehicle Routing Problem (VRP) into a QUBO model.
        :param vehicle_count: Number of available vehicles.
        """
        num_nodes = len(self.graph.nodes)
        depot = 0  # Assume node 0 is the depot
        x = {(i, t): f"x_{i}_{t}" for i in range(num_nodes) for t in range(num_nodes)}
        penalty = self.adaptive_penalty_factor()

        # Objective: Minimize total travel cost
        for i, j in permutations(range(num_nodes), 2):
            weight = self.graph[i][j]['weight']
            for t in range(num_nodes - 1):
                self.bqm.add_interaction(x[(i, t)], x[(j, t + 1)], weight)

        # Constraint: Each location is visited at most once
        for i in range(1, num_nodes):  # Exclude depot
            constraint = sum(self.bqm.add_variable(x[(i, t)], -penalty) for t in range(num_nodes))
            self.bqm.add_offset(penalty * (constraint ** 2))

        # Constraint: Each vehicle starts and ends at the depot
        for k in range(vehicle_count):
            self.bqm.add_variable(x[(depot, 0)], -penalty)
            self.bqm.add_variable(x[(depot, num_nodes - 1)], -penalty)

        return self.bqm

    def knapsack_qubo(self, values, weights, capacity):
        """
        Converts the Knapsack Problem into a QUBO formulation.
        :param values: List of item values.
        :param weights: List of item weights.
        :param capacity: Maximum weight allowed.
        """
        num_items = len(values)
        x = {i: f"x_{i}" for i in range(num_items)}
        penalty = self.adaptive_penalty_factor()

        # Objective: Maximize total value
        for i in range(num_items):
            self.bqm.add_variable(x[i], -values[i])

        # Constraint: Total weight should not exceed capacity
        weight_constraint = sum(weights[i] * self.bqm.add_variable(x[i], -penalty) for i in range(num_items))
        self.bqm.add_offset(penalty * (max(0, weight_constraint - capacity) ** 2))

        return self.bqm

    def generate_qubo(self):
        """
        Returns the appropriate QUBO model based on the selected problem type.
        """
        if self.problem_type == "TSP":
            return self.tsp_qubo()
        elif self.problem_type == "VRP":
            return self.vrp_qubo()
        elif self.problem_type == "Knapsack":
            return self.knapsack_qubo()
        else:
            raise ValueError("Unsupported problem type. Choose 'TSP', 'VRP', or 'Knapsack'.")

if __name__ == "__main__":
    # Example: TSP Graph with 4 Cities
    G = nx.Graph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(1, 2, weight=15)
    G.add_edge(2, 3, weight=20)
    G.add_edge(3, 0, weight=25)

    tsp_solver = QUBOFormulation(graph=G, problem_type="TSP")
    tsp_solver.preprocess_graph()
    tsp_qubo = tsp_solver.generate_qubo()

    print("\n🔍 **Generated QUBO for TSP:**")
    print(tsp_qubo)