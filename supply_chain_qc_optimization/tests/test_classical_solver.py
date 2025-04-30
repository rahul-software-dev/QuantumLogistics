import unittest
import numpy as np
from src.optimization.classical_solver import SimulatedAnnealingSolver
from src.optimization.problems.tsp_problem import TSPProblem

class TestClassicalSolver(unittest.TestCase):
    def setUp(self):
        self.solver = SimulatedAnnealingSolver(max_iter=100)
        self.distance_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        self.problem = TSPProblem(self.distance_matrix)

    def test_solve_tsp(self):
        route, cost = self.solver.solve_tsp(self.problem)
        self.assertEqual(len(route), self.problem.num_cities)
        self.assertTrue(cost >= 0)

if __name__ == '__main__':
    unittest.main()
