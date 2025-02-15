import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

class SupplyChainVisualizer:
    """
    Visualizes optimized supply chain routes from classical & quantum solvers.
    Supports static, animated, and interactive visualization.
    """

    def __init__(self, data_file="supply_chain_data.csv"):
        """
        Initializes visualization with logistics data.
        """
        self.df = pd.read_csv(data_file)
        self.graph = nx.Graph()
        self.geolocator = Nominatim(user_agent="supply_chain")

    def build_graph(self):
        """
        Constructs a network graph from CSV data (Node = City, Edge = Route).
        """
        for _, row in self.df.iterrows():
            self.graph.add_edge(row['Source'], row['Destination'], weight=row['Distance'])

    def visualize_static_routes(self):
        """
        Displays a static visualization of the supply chain network.
        """
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(10, 6))
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color="skyblue", edge_color="gray")
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)
        plt.title("Optimized Supply Chain Routes")
        plt.show()

    def visualize_interactive_routes(self):
        """
        Creates an interactive route visualization using Plotly.
        """
        fig = go.Figure()
        for edge in self.graph.edges(data=True):
            src, dest, attr = edge
            fig.add_trace(go.Scatter(
                x=[self.df.loc[self.df["Source"] == src, "Longitude"].values[0],
                   self.df.loc[self.df["Destination"] == dest, "Longitude"].values[0]],
                y=[self.df.loc[self.df["Source"] == src, "Latitude"].values[0],
                   self.df.loc[self.df["Destination"] == dest, "Latitude"].values[0]],
                mode="lines",
                line=dict(width=2, color="blue"),
                name=f"{src} → {dest}"
            ))

        fig.update_layout(title="Optimized Supply Chain Network", xaxis_title="Longitude", yaxis_title="Latitude")
        fig.show()

    def animate_shipment(self, route):
        """
        Animates a shipment moving through an optimized route.
        """
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color="skyblue")

        for i in range(len(route) - 1):
            nx.draw_networkx_edges(self.graph, pos, edgelist=[(route[i], route[i + 1])], edge_color="red", width=2)
            plt.pause(0.5)

        plt.title("Shipment Movement Through Supply Chain")
        plt.show()

if __name__ == "__main__":
    visualizer = SupplyChainVisualizer()
    visualizer.build_graph()
    visualizer.visualize_static_routes()
    visualizer.visualize_interactive_routes()
    visualizer.animate_shipment(["Mumbai", "Delhi", "Chennai"])