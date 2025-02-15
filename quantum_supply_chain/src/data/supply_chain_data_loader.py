import pandas as pd
import numpy as np
import networkx as nx
import json
import requests
from typing import Dict, Tuple, List

class SupplyChainDataLoader:
    """
    Loads and preprocesses supply chain data for quantum and classical logistics optimization.
    Supports multiple data formats (CSV, JSON, API) and encodes it into graphs/matrices.
    """

    def __init__(self, source_type: str, source_path: str = None, api_url: str = None):
        """
        Initializes the data loader with the source type.
        :param source_type: "csv", "json", or "api"
        :param source_path: File path for CSV/JSON data
        :param api_url: API URL for real-time data fetching
        """
        self.source_type = source_type
        self.source_path = source_path
        self.api_url = api_url
        self.data = None
        self.graph = None  # Graph representation for network-based problems

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from CSV, JSON, or API.
        :return: Pandas DataFrame of supply chain data
        """
        if self.source_type == "csv":
            self.data = pd.read_csv(self.source_path)
        elif self.source_type == "json":
            with open(self.source_path, 'r') as file:
                self.data = pd.DataFrame(json.load(file))
        elif self.source_type == "api":
            response = requests.get(self.api_url)
            if response.status_code == 200:
                self.data = pd.DataFrame(response.json())
            else:
                raise Exception(f"API request failed with status code {response.status_code}")
        else:
            raise ValueError("Invalid source type. Use 'csv', 'json', or 'api'.")

        self.clean_data()
        return self.data

    def clean_data(self):
        """
        Cleans and normalizes the data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Remove duplicates and handle missing values
        self.data.drop_duplicates(inplace=True)
        self.data.fillna(method="ffill", inplace=True)

        # Normalize latitude & longitude if present
        if {"latitude", "longitude"}.issubset(self.data.columns):
            self.data["latitude"] = (self.data["latitude"] - self.data["latitude"].min()) / (
                    self.data["latitude"].max() - self.data["latitude"].min())
            self.data["longitude"] = (self.data["longitude"] - self.data["longitude"].min()) / (
                    self.data["longitude"].max() - self.data["longitude"].min())

        print("✔ Data cleaning complete.")

    def generate_distance_matrix(self) -> np.ndarray:
        """
        Converts supply chain locations into a distance matrix for logistics optimization.
        :return: Numpy array representing the distance matrix
        """
        if {"latitude", "longitude"}.issubset(self.data.columns):
            locations = self.data[["latitude", "longitude"]].values
            distance_matrix = np.linalg.norm(locations[:, np.newaxis] - locations, axis=2)
            return distance_matrix
        else:
            raise ValueError("Latitude and longitude columns required for distance matrix.")

    def build_graph(self) -> nx.Graph:
        """
        Constructs a graph representation of the supply chain.
        :return: NetworkX Graph object
        """
        self.graph = nx.Graph()

        for _, row in self.data.iterrows():
            self.graph.add_node(row["location_id"], pos=(row["latitude"], row["longitude"]))

        # Add weighted edges (distances) between nodes
        for i, row1 in self.data.iterrows():
            for j, row2 in self.data.iterrows():
                if i != j:
                    distance = np.linalg.norm(
                        np.array([row1["latitude"], row1["longitude"]]) - 
                        np.array([row2["latitude"], row2["longitude"]])
                    )
                    self.graph.add_edge(row1["location_id"], row2["location_id"], weight=distance)

        return self.graph

    def encode_for_quantum(self) -> Tuple[np.ndarray, Dict[int, Tuple[float, float]]]:
        """
        Encodes the graph into a format suitable for quantum solvers.
        :return: (Adjacency matrix, Node position dictionary)
        """
        if self.graph is None:
            self.build_graph()

        adjacency_matrix = nx.to_numpy_matrix(self.graph)
        node_positions = {node: data["pos"] for node, data in self.graph.nodes(data=True)}

        return adjacency_matrix, node_positions


# Example Usage
if __name__ == "__main__":
    # Load from CSV
    data_loader = SupplyChainDataLoader(source_type="csv", source_path="data/supply_chain.csv")
    df = data_loader.load_data()
    print("Loaded Data:\n", df.head())

    # Generate Distance Matrix
    distance_matrix = data_loader.generate_distance_matrix()
    print("Distance Matrix:\n", distance_matrix)

    # Build Graph Representation
    graph = data_loader.build_graph()
    print("Graph Nodes:", graph.nodes)
    print("Graph Edges:", graph.edges(data=True))

    # Encode Data for Quantum Computing
    adjacency_matrix, positions = data_loader.encode_for_quantum()
    print("Quantum Adjacency Matrix:\n", adjacency_matrix)