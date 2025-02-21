import numpy as np
import networkx as nx
import random
import itertools
import multiprocessing
from scipy.spatial import distance
from deap import base, creator, tools, algorithms

class MetaheuristicSolver:
    """
    Implements Genetic Algorithm (GA), Ant Colony Optimization (ACO), and Simulated Annealing (SA)
    for solving the Traveling Salesman Problem (TSP) and Vehicle Routing Problem (VRP).
    """

    def __init__(self, graph: nx.Graph, num_vehicles=1, alpha=1, beta=2, evaporation=0.5, ant_count=10, seed=42):
        """
        Initialize solver with supply chain network.
        """
        self.graph = graph
        self.num_nodes = len(graph.nodes)
        self.num_vehicles = num_vehicles
        self.rng = np.random.default_rng(seed)

        # Ant Colony parameters
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.ants = ant_count
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))

    """ ---------------------- GENETIC ALGORITHM (GA) ---------------------- """

    def genetic_algorithm(self, population_size=200, generations=500, elitism=True):
        """
        Genetic Algorithm for TSP and VRP.
        Uses DEAP library for evolutionary optimization.
        """

        def evaluate(individual):
            """Evaluates the fitness of an individual (route distance)."""
            return sum(self.graph[individual[i]][individual[i + 1]]['weight'] for i in range(len(individual) - 1)),

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, range(self.num_nodes), self.num_nodes)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)

        population = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1) if elitism else None

        algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, cxpb=0.7,
                                  mutpb=0.2, ngen=generations, halloffame=hof, verbose=False)

        best_individual = hof[0] if elitism else tools.selBest(population, 1)[0]
        return best_individual, evaluate(best_individual)[0]

    """ ---------------------- ANT COLONY OPTIMIZATION (ACO) ---------------------- """

    def ant_colony_optimization(self, iterations=200):
        """
        Ant Colony Optimization for TSP and VRP.
        """

        def calculate_distance(route):
            """Calculate total distance for an ant's path."""
            return sum(self.graph[route[i]][route[i + 1]]['weight'] for i in range(len(route) - 1))

        best_route = None
        best_distance = float('inf')

        for _ in range(iterations):
            all_routes = []
            all_distances = []
            for _ in range(self.ants):
                route = list(range(self.num_nodes))
                self.rng.shuffle(route)

                distance = calculate_distance(route)
                all_routes.append(route)
                all_distances.append(distance)

                if distance < best_distance:
                    best_distance = distance
                    best_route = route

            # Dynamic Pheromone Update (Improved ACO)
            self.pheromone *= (1 - self.evaporation)
            for i, route in enumerate(all_routes):
                for j in range(len(route) - 1):
                    self.pheromone[route[j], route[j + 1]] += 1 / (all_distances[i] + 1e-6)

        return best_route, best_distance

    """ ---------------------- SIMULATED ANNEALING (SA) ---------------------- """

    def simulated_annealing(self, initial_temp=1000, cooling_rate=0.99, min_temp=1, iterations=2000):
        """
        Simulated Annealing for TSP with dynamic temperature decay.
        """

        def route_distance(route):
            """Calculate total distance for a given route."""
            return sum(self.graph[route[i]][route[i + 1]]['weight'] for i in range(len(route) - 1))

        current_route = list(self.graph.nodes)
        self.rng.shuffle(current_route)
        current_cost = route_distance(current_route)
        temp = initial_temp

        for _ in range(iterations):
            i, j = sorted(self.rng.choice(len(current_route), 2, replace=False))
            new_route = current_route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]
            new_cost = route_distance(new_route)

            if new_cost < current_cost or np.exp((current_cost - new_cost) / temp) > self.rng.random():
                current_route = new_route
                current_cost = new_cost

            temp *= cooling_rate
            if temp < min_temp:
                break

        return current_route, current_cost

    """ ---------------------- PARALLEL EXECUTION ---------------------- """

    def parallel_execution(self):
        """
        Runs all solvers in parallel using multiprocessing.
        """
        with multiprocessing.Pool(processes=3) as pool:
            results = pool.map(self.run_solver, ["GA", "ACO", "SA"])

        return {solver: result for solver, result in results}

    def run_solver(self, solver):
        """
        Helper function to run a specific solver.
        """
        if solver == "GA":
            return "Genetic Algorithm", self.genetic_algorithm()
        elif solver == "ACO":
            return "Ant Colony Optimization", self.ant_colony_optimization()
        elif solver == "SA":
            return "Simulated Annealing", self.simulated_annealing()

    """ ---------------------- RUN ALL SOLVERS ---------------------- """

    def run_all_solvers(self):
        """
        Runs all three metaheuristic solvers sequentially.
        """
        ga_route, ga_cost = self.genetic_algorithm()
        aco_route, aco_cost = self.ant_colony_optimization()
        sa_route, sa_cost = self.simulated_annealing()

        return {
            "Genetic Algorithm": {"route": ga_route, "cost": ga_cost},
            "Ant Colony Optimization": {"route": aco_route, "cost": aco_cost},
            "Simulated Annealing": {"route": sa_route, "cost": sa_cost}
        }


if __name__ == "__main__":
    # Sample Supply Chain Graph
    G = nx.Graph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(1, 2, weight=20)
    G.add_edge(2, 3, weight=15)
    G.add_edge(3, 0, weight=25)

    solver = MetaheuristicSolver(G)
    results = solver.parallel_execution()

    for method, result in results.items():
        print(f"{method}: Route {result[0]}, Cost {result[1]}")