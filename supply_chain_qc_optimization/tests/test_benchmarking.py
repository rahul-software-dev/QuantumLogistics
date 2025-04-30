import unittest
import numpy as np
from src.benchmarking.benchmarking import benchmark_solver
from src.optimization.classical_solver import SimulatedAnnealingSolver
from src.optimization.problems.tsp_problem import TSPProblem

class TestBenchmarking(unittest.TestCase):
    def setUp(self):
        self.solver = SimulatedAnnealingSolver(max_iter=100)
        self.distance_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        self.problem = TSPProblem(self.distance_matrix)

    def test_benchmark_solver(self):
        result = benchmark_solver(self.solver, self.problem)
        self.assertIn('solution', result.as_dict())
        self.assertIn('objective_value', result.as_dict())

if __name__ == '__main__':
    unittest.main()
