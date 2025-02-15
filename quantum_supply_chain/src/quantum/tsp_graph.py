import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests

class SupplyChainGraph:
    def __init__(self, locations):
        """
        Initializes the supply chain graph with given warehouse locations.
        """
        self.locations = locations
        self.num_locations = len(locations)
        self.G = nx.complete_graph(self.num_locations)  # Fully connected TSP graph
        self.distances = self.get_real_distances()  # Get real-world distances
        self.assign_weights()  # Assign distances as edge weights

    def get_real_distances(self):
        """
        Uses Google Maps API to fetch real-world distances (in km) between warehouse locations.
        """
        distances = np.zeros((self.num_locations, self.num_locations))

        for i in range(self.num_locations):
            for j in range(i + 1, self.num_locations):
                # Fetch real-world distance using an API (e.g., Google Maps API)
                distance = self.get_distance_from_api(self.locations[i], self.locations[j])
                distances[i][j] = distances[j][i] = distance

        return distances

    @staticmethod
    def get_distance_from_api(origin, destination):
        """
        Fetches distance between two locations using an external API.
        Replace 'YOUR_API_KEY' with an actual Google Maps or OpenRoute API key.
        """
        API_KEY = "YOUR_API_KEY"  # Replace with your own key
        url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin}&destinations={destination}&key={API_KEY}"

        response = requests.get(url).json()
        if response["status"] == "OK":
            return response["rows"][0]["elements"][0]["distance"]["value"] / 1000  # Convert meters to km
        return np.random.randint(5, 50)  # Fallback to random if API fails

    def assign_weights(self):
        """
        Assigns real-world distances as weights to the graph edges.
        """
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = self.distances[i][j]

    def visualize_graph(self):
        """
        Visualizes the supply chain graph.
        """
        pos = nx.spring_layout(self.G, seed=42)
        labels = nx.get_edge_attributes(self.G, 'weight')

        nx.draw(self.G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels={k: f"{v:.1f} km" for k, v in labels.items()})

        plt.title("Real-World Supply Chain Network (TSP)")
        plt.show()

    def export_to_csv(self, filename="supply_chain_data.csv"):
        """
        Exports the distance matrix to a CSV file.
        """
        df = pd.DataFrame(self.distances, columns=self.locations, index=self.locations)
        df.to_csv(filename)
        print(f"Distance matrix saved to {filename}")

# Example Usage
if __name__ == "__main__":
    warehouse_locations = [
        "New York, USA",
        "Los Angeles, USA",
        "Chicago, USA",
        "Houston, USA",
        "Miami, USA"
    ]

    tsp_graph = SupplyChainGraph(warehouse_locations)
    tsp_graph.visualize_graph()
    tsp_graph.export_to_csv()