import numpy as np
import pandas as pd
import random
from itertools import permutations
from quantum_supply_chain.src.quantum.tsp_qubo import TSPQUBO
from multiprocessing import Pool

class ClassicalTSP:
    def __init__(self, csv_file="data/supply_chain_data.csv", population_size=100, generations=500):
        """
        Initializes the TSP solver with real-world data.
        """
        self.tsp = TSPQUBO(csv_file)  # Load distance matrix from real-world data
        self.locations = self.tsp.locations
        self.dist_matrix = self.tsp.dist_matrix
        self.num_cities = len(self.locations)
        self.population_size = population_size
        self.generations = generations
    
    def compute_distance(self, path):
        """
        Computes total distance for a given TSP route.
        """
        return sum(self.dist_matrix[path[i], path[i+1]] for i in range(len(path) - 1)) + self.dist_matrix[path[-1], path[0]]
    
    # ============================== GENETIC ALGORITHM ==============================
    def genetic_algorithm(self):
        """
        Solves TSP using Genetic Algorithm.
        """
        def create_population():
            return [random.sample(range(self.num_cities), self.num_cities) for _ in range(self.population_size)]
        
        def crossover(parent1, parent2):
            split = random.randint(0, self.num_cities - 1)
            child = parent1[:split] + [city for city in parent2 if city not in parent1[:split]]
            return child
        
        def mutate(route, mutation_rate=0.05):
            if random.random() < mutation_rate:
                i, j = random.sample(range(self.num_cities), 2)
                route[i], route[j] = route[j], route[i]
            return route
        
        def fitness(route):
            return 1 / self.compute_distance(route)

        population = create_population()
        for _ in range(self.generations):
            population = sorted(population, key=lambda x: self.compute_distance(x))
            new_population = population[:10]  # Keep best 10 solutions
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(population[:50], 2)  # Select top 50
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            population = new_population
        
        best_route = min(population, key=lambda x: self.compute_distance(x))
        return best_route, self.compute_distance(best_route)

    # =========================== SIMULATED ANNEALING ===========================
    def simulated_annealing(self, initial_temp=1000, cooling_rate=0.99):
        """
        Solves TSP using Simulated Annealing.
        """
        def swap(route):
            i, j = random.sample(range(self.num_cities), 2)
            route[i], route[j] = route[j], route[i]
            return route
        
        current_route = random.sample(range(self.num_cities), self.num_cities)
        current_cost = self.compute_distance(current_route)
        temp = initial_temp

        while temp > 1:
            new_route = swap(current_route[:])
            new_cost = self.compute_distance(new_route)
            if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temp):
                current_route, current_cost = new_route, new_cost
            temp *= cooling_rate

        return current_route, current_cost

    # =========================== TABU SEARCH ===========================
    def tabu_search(self, tabu_size=100, max_iter=1000):
        """
        Solves TSP using Tabu Search.
        """
        current_route = random.sample(range(self.num_cities), self.num_cities)
        current_cost = self.compute_distance(current_route)
        best_route, best_cost = current_route, current_cost
        tabu_list = []

        for _ in range(max_iter):
            neighbors = [current_route[:i] + current_route[i+1:j] + [current_route[i]] + current_route[j:] for i in range(self.num_cities) for j in range(i+1, self.num_cities)]
            neighbors = sorted(neighbors, key=lambda x: self.compute_distance(x))
            
            for candidate in neighbors:
                if candidate not in tabu_list:
                    current_route = candidate
                    current_cost = self.compute_distance(candidate)
                    if current_cost < best_cost:
                        best_route, best_cost = candidate, current_cost
                    tabu_list.append(candidate)
                    if len(tabu_list) > tabu_size:
                        tabu_list.pop(0)
                    break
        
        return best_route, best_cost

    # =========================== MULTITHREADING FOR SPEED ===========================
    def solve_parallel(self):
        """
        Runs Genetic Algorithm, Simulated Annealing, and Tabu Search in parallel.
        """
        with Pool(3) as pool:
            results = pool.starmap_async(self._solve_method, [(self.genetic_algorithm, ), (self.simulated_annealing, ), (self.tabu_search, )]).get()
        
        return results

    def _solve_method(self, method):
        return method()

# =========================== RUN & BENCHMARK ===========================
if __name__ == "__main__":
    solver = ClassicalTSP()
    
    # Solve using Genetic Algorithm
    ga_solution, ga_cost = solver.genetic_algorithm()
    print("\n🧬 Genetic Algorithm Solution:")
    print("Route:", ga_solution, "Cost:", ga_cost)
    
    # Solve using Simulated Annealing
    sa_solution, sa_cost = solver.simulated_annealing()
    print("\n🔥 Simulated Annealing Solution:")
    print("Route:", sa_solution, "Cost:", sa_cost)

    # Solve using Tabu Search
    ts_solution, ts_cost = solver.tabu_search()
    print("\n🚫 Tabu Search Solution:")
    print("Route:", ts_solution, "Cost:", ts_cost)

    # Benchmark with QAOA Solver
    from quantum_supply_chain.src.quantum.qaoa_solver import QAOASolver
    qaoa_solver = QAOASolver(p=3)
    qaoa_result = qaoa_solver.solve_with_qaoa()
    print("\n⚛️ QAOA Quantum Solution:")
    print(qaoa_result)