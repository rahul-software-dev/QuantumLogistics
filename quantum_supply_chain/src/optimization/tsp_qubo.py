import numpy as np
import pandas as pd
from itertools import permutations
from qiskit.optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model

class TSPQUBO:
    def __init__(self, csv_file="data/supply_chain_data.csv"):
        """
        Initializes the QUBO model for TSP using real-world distances.
        """
        self.locations, self.dist_matrix = self.load_distance_matrix(csv_file)
        self.num_cities = len(self.locations)
        self.qubo_model = None  # QUBO Model (to be generated)
        
    def load_distance_matrix(self, csv_file):
        """
        Loads the real-world distance matrix from a CSV file.
        """
        df = pd.read_csv(csv_file, index_col=0)
        locations = df.columns.tolist()
        return locations, df.values  # Return location names & numpy distance matrix

    def tsp_qubo_formulation(self):
        """
        Formulates the TSP problem as a QUBO using Docplex.
        """
        mdl = Model(name="TSP_QUBO")
        x = {(i, j): mdl.binary_var(name=f"x_{i}_{j}") for i in range(self.num_cities) for j in range(self.num_cities)}

        # **1. Objective Function: Minimize total travel cost**
        mdl.minimize(mdl.sum(self.dist_matrix[i, j] * x[i, j] for i in range(self.num_cities) for j in range(self.num_cities) if i != j))

        # **2. Constraint: Each city must be visited exactly once**
        for i in range(self.num_cities):
            mdl.add_constraint(mdl.sum(x[i, j] for j in range(self.num_cities) if i != j) == 1)  # Leave city once
            mdl.add_constraint(mdl.sum(x[j, i] for j in range(self.num_cities) if i != j) == 1)  # Enter city once

        # **3. Constraint: Prevent sub-tours (Miller-Tucker-Zemlin Constraint)**
        u = {i: mdl.continuous_var(lb=0, name=f"u_{i}") for i in range(1, self.num_cities)}
        for i in range(1, self.num_cities):
            for j in range(1, self.num_cities):
                if i != j:
                    mdl.add_constraint(u[i] - u[j] + self.num_cities * x[i, j] <= self.num_cities - 1)

        # Convert to QUBO format
        self.qubo_model = from_docplex_mp(mdl)

    def get_qubo_model(self):
        """
        Returns the formulated QUBO model.
        """
        if self.qubo_model is None:
            self.tsp_qubo_formulation()
        return self.qubo_model

    def save_qubo_to_file(self, filename="data/tsp_qubo.lp"):
        """
        Saves the QUBO model to a file for quantum solvers.
        """
        if self.qubo_model is None:
            self.tsp_qubo_formulation()
        with open(filename, "w") as f:
            f.write(str(self.qubo_model.export_as_lp_string()))
        print(f"QUBO model saved to {filename}")

# Example Usage
if __name__ == "__main__":
    tsp_qubo = TSPQUBO()
    qubo_model = tsp_qubo.get_qubo_model()
    tsp_qubo.save_qubo_to_file()