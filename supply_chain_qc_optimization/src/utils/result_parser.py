"""
Result Parsing and Formatting Utilities
--------------------------------------
Parse and format solutions from quantum/classical solvers for reporting and analysis.
"""
import numpy as np

def parse_tsp_solution(bitstring, n):
    """
    Parses a TSP QUBO bitstring into a route.
    """
    assignment = np.array([int(b) for b in bitstring]).reshape((n, n))
    route = assignment.argmax(axis=0)
    return route.tolist()

def format_route(route, locations=None):
    """
    Formats a route as a human-readable string.
    """
    if locations:
        return " -> ".join([locations[i] for i in route] + [locations[route[0]]])
    return " -> ".join(map(str, route + [route[0]]))

def summarize_results(result_dict):
    """
    Summarizes benchmarking or solver results in a readable format.
    """
    summary = (
        f"Solver: {result_dict.get('solver')}\n"
        f"Problem: {result_dict.get('problem')}\n"
        f"Objective Value: {result_dict.get('objective_value')}\n"
        f"Runtime: {result_dict.get('runtime'):.4f} seconds\n"
        f"Solution: {result_dict.get('solution')}\n"
    )
    return summary
