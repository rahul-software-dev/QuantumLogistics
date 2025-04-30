import time
import numpy as np

class BenchmarkResult:
    def __init__(self, solver_name, problem_name, solution, objective_value, runtime, metadata=None):
        self.solver_name = solver_name
        self.problem_name = problem_name
        self.solution = solution
        self.objective_value = objective_value
        self.runtime = runtime
        self.metadata = metadata or {}

    def as_dict(self):
        return {
            "solver": self.solver_name,
            "problem": self.problem_name,
            "objective_value": self.objective_value,
            "runtime": self.runtime,
            "solution": self.solution,
            "metadata": self.metadata
        }

def benchmark_solver(solver, problem, solver_name=None, problem_name=None, **solver_kwargs):
    """
    Benchmarks a solver on a given problem instance.
    :param solver: Callable or object with a .solve() method
    :param problem: Problem instance (e.g., TSPProblem)
    :param solver_name: Optional name for the solver
    :param problem_name: Optional name for the problem
    :param solver_kwargs: Additional kwargs for the solver
    :return: BenchmarkResult
    """
    start = time.time()
    # If the solver is a class instance with a .solve() method, use it
    if callable(getattr(solver, "solve", None)):
        solution, objective_value = solver.solve(problem, **solver_kwargs)
    elif callable(solver):
        solution, objective_value = solver(problem, **solver_kwargs)
    else:
        raise TypeError(f"Solver {solver} is neither callable nor has a .solve() method.")
    end = time.time()
    return BenchmarkResult(
        solver_name=solver_name or solver.__class__.__name__,
        problem_name=problem_name or problem.__class__.__name__,
        solution=solution,
        objective_value=objective_value,
        runtime=end - start
    )
