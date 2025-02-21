import numpy as np
import networkx as nx
from qiskit import Aer
from qiskit.algorithms import QAOA, VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp
from deap import base, creator, tools, algorithms  # Genetic Algorithm library
from itertools import permutations
import multiprocessing
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class HybridMetaheuristicQuantumSolver:
    """
    Advanced Hybrid Metaheuristic + Quantum Optimization for Supply Chain Routing.
    Uses Genetic Algorithm (GA), Ant Colony Optimization (ACO), and Quantum Solvers (QAOA/VQE).
    """

    def __init__(self, graph: nx.Graph, num_nodes: int, method="GA", quantum_algorithm="QAOA"):
        """
        Initializes the solver with a supply chain graph.
        - method: "GA" (Genetic Algorithm) or "ACO" (Ant Colony Optimization)
        - quantum_algorithm: "QAOA" or "VQE"
        """
        self.graph = graph
        self.num_nodes = num_nodes
        self.method = method
        self.quantum_algorithm = quantum_algorithm
        self.best_classical_solution = None
        self.best_quantum_solution = None

    ### === 1. Classical Metaheuristic Optimization === ###

    def fitness_function(self, route):
        """Fitness function for evaluating supply chain routes."""
        return sum(self.graph[route[i]][route[i + 1]]['weight'] for i in range(len(route) - 1))

    def genetic_algorithm(self, population_size=100, generations=150):
        """Solves TSP/VRP using Genetic Algorithm (GA) with parallelism."""

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", np.random.permutation, self.num_nodes)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.fitness_function)

        population = toolbox.population(n=population_size)

        # Parallel Computing for Speedup
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        logging.info("[Genetic Algorithm] Running Evolutionary Process...")
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, verbose=True)

        best_individual = tools.selBest(population, 1)[0]
        self.best_classical_solution = best_individual
        return best_individual

    def ant_colony_optimization(self, num_ants=50, num_iterations=100, alpha=1, beta=2, evaporation=0.5):
        """Solves TSP/VRP using Ant Colony Optimization (ACO)."""
        pheromones = {edge: 1.0 for edge in self.graph.edges}

        for iteration in range(num_iterations):
            routes = []
            for _ in range(num_ants):
                route = list(np.random.permutation(self.num_nodes))
                routes.append(route)

            best_route = min(routes, key=self.fitness_function)

            # Update Pheromones
            for edge in self.graph.edges:
                pheromones[edge] *= (1 - evaporation)  # Evaporation
                if edge in zip(best_route, best_route[1:]):
                    pheromones[edge] += 1.0 / self.fitness_function(best_route)

        self.best_classical_solution = best_route
        return best_route

    ### === 2. Quantum Optimization with QAOA/VQE === ###

    def get_qubo_matrix(self):
        """Generates QUBO formulation for Quantum Optimization."""
        qubo = {}
        for i, j in permutations(range(self.num_nodes), 2):
            qubo[(i, j)] = self.graph[i][j]['weight']
        return qubo

    def quantum_optimizer(self):
        """
        Solves the problem using Quantum Optimization (QAOA or VQE).
        """
        pauli_op = PauliSumOp.from_list([("ZZ", weight) for (i, j), weight in self.get_qubo_matrix().items()])

        backend = Aer.get_backend('statevector_simulator')
        
        if self.quantum_algorithm == "QAOA":
            optimizer = QAOA(optimizer=COBYLA(), reps=3, quantum_instance=backend)
        else:
            optimizer = VQE(ansatz=None, optimizer=COBYLA(), quantum_instance=backend)  # Define ansatz as needed
        
        logging.info(f"[Quantum {self.quantum_algorithm}] Running Quantum Optimization...")
        result = optimizer.compute_minimum_eigenvalue(operator=pauli_op)
        self.best_quantum_solution = result.eigenvalue.real

        return self.best_quantum_solution

    ### === 3. Hybrid Quantum-Classical Iterative Refinement === ###
    
    def hybrid_solver(self, hybrid_iterations=5):
        """
        Runs Hybrid Metaheuristic + Quantum Optimization iteratively.
        """
        best_solution = None

        for i in range(hybrid_iterations):
            logging.info(f"\n[Iteration {i+1}] Running Metaheuristic Optimization...")
            if self.method == "GA":
                classical_solution = self.genetic_algorithm()
            else:
                classical_solution = self.ant_colony_optimization()

            logging.info("[Quantum Refinement] Running Quantum Optimization...")
            quantum_solution = self.quantum_optimizer()

            if best_solution is None or quantum_solution < best_solution:
                best_solution = quantum_solution

            logging.info(f"[Iteration {i+1}]: Best Hybrid Solution: {best_solution}")

        return best_solution

if __name__ == "__main__":
    # Sample Supply Chain Graph
    G = nx.Graph()
    G.add_edge(0, 1, weight=12)
    G.add_edge(1, 2, weight=15)
    G.add_edge(2, 3, weight=10)
    G.add_edge(3, 0, weight=20)
    G.add_edge(1, 3, weight=18)

    solver = HybridMetaheuristicQuantumSolver(graph=G, num_nodes=4, method="GA", quantum_algorithm="QAOA")
    best_hybrid_solution = solver.hybrid_solver()

    logging.info(f"\nFinal Best Hybrid Solution: {best_hybrid_solution}")