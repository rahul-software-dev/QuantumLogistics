"""
Graph-Based Helper Functions
----------------------------
Utilities for graph construction, manipulation, and analysis in supply chain problems.
References:
- NetworkX Documentation: https://networkx.org/documentation/stable/
"""
import networkx as nx
import numpy as np

def distance_matrix_to_graph(distance_matrix, locations=None):
    """
    Converts a distance matrix to a NetworkX graph.
    """
    n = distance_matrix.shape[0]
    G = nx.Graph()
    if locations is None:
        locations = [f"Node_{i}" for i in range(n)]
    for i in range(n):
        G.add_node(locations[i])
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] > 0:
                G.add_edge(locations[i], locations[j], weight=distance_matrix[i, j])
    return G

def route_length(route, distance_matrix):
    """
    Computes the total length of a given route.
    """
    return sum(distance_matrix[route[i-1], route[i]] for i in range(len(route)))

def get_shortest_path(graph, source, target, weight='weight'):
    """
    Returns the shortest path and its length between source and target.
    """
    path = nx.shortest_path(graph, source=source, target=target, weight=weight)
    length = nx.shortest_path_length(graph, source=source, target=target, weight=weight)
    return path, length
