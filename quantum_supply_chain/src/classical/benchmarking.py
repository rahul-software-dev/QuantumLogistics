import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool

# Import Classical Solvers
from quantum_supply_chain.src.classical.classical_solver import ClassicalTSP

# Import Quantum Solvers
from quantum_supply_chain.src.optimization.qaoa_solver import QAOASolver
from quantum_supply_chain.src.optimization.tsp_qubo import TSPQUBO

class BenchmarkTSP:
    def __init__(self):
        """
        Initializes benchmark tests for classical and quantum solvers.
        """
        self.classical_solver = ClassicalTSP()
        self.qaoa_solver = QAOASolver(p=3)
        self.qubo_solver = TSPQUBO()

        # Results storage
        self.results = []

    def benchmark_solver(self, solver_method, solver_name):
        """
        Runs a solver and records execution time & cost.
        """
        start_time = time.time()
        solution, cost = solver_method()
        execution_time = time.time() - start_time

        self.results.append({
            "Solver": solver_name,
            "Execution Time (s)": execution_time,
            "Cost (Distance)": cost
        })

        return solution, cost

    def run_benchmarks(self):
        """
        Executes benchmarking on all solvers.
        """
        solvers = [
            (self.classical_solver.genetic_algorithm, "Genetic Algorithm"),
            (self.classical_solver.simulated_annealing, "Simulated Annealing"),
            (self.classical_solver.tabu_search, "Tabu Search"),
            (self.qaoa_solver.solve_with_qaoa, "Quantum QAOA"),
            (self.qubo_solver.solve_qubo, "Quantum Annealing")
        ]

        with Pool(len(solvers)) as pool:
            results = pool.starmap(self.benchmark_solver, solvers)

        return results

    def plot_results(self):
        """
        Visualizes execution time & cost for all solvers.
        """
        df = pd.DataFrame(self.results)

        # Execution Time Plot
        plt.figure(figsize=(10, 5))
        sns.barplot(x="Solver", y="Execution Time (s)", data=df, palette="coolwarm")
        plt.title("Execution Time Comparison")
        plt.xticks(rotation=45)
        plt.show()

        # Cost Comparison Plot
        plt.figure(figsize=(10, 5))
        sns.barplot(x="Solver", y="Cost (Distance)", data=df, palette="coolwarm")
        plt.title("TSP Solution Cost Comparison")
        plt.xticks(rotation=45)
        plt.show()

    def display_results(self):
        """
        Displays benchmark results in tabular format.
        """
        df = pd.DataFrame(self.results)
        print("\n📊 **Benchmark Results:**")
        print(df.to_string(index=False))

if __name__ == "__main__":
    benchmark = BenchmarkTSP()
    benchmark.run_benchmarks()
    benchmark.display_results()
    benchmark.plot_results()