import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt
from qiskit.quantum_info import state_fidelity

class QuantumGateNoise:
    """
    Simulates Quantum Gate Noise using Qiskit.
    Models depolarizing, thermal relaxation, and readout noise.
    """

    def __init__(self):
        print("🔬 Initializing Quantum Noise Simulation...")
        self.noise_model = NoiseModel()

        # Define error rates
        self.depolarizing_prob = 0.01  # 1% chance of depolarization
        self.thermal_t1 = 50e3  # T1 relaxation time in microseconds
        self.thermal_t2 = 30e3  # T2 dephasing time in microseconds
        self.readout_error_prob = 0.02  # 2% readout error

        self.apply_depolarizing_noise()
        self.apply_thermal_relaxation()
        self.apply_readout_error()

    def apply_depolarizing_noise(self):
        """Adds depolarizing noise to single and two-qubit gates."""
        error_1q = depolarizing_error(self.depolarizing_prob, 1)
        error_2q = depolarizing_error(self.depolarizing_prob * 3, 2)  # Higher for 2-qubit gates

        self.noise_model.add_all_qubit_quantum_error(error_1q, ['h', 't', 'x', 'y', 'z'])
        self.noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    def apply_thermal_relaxation(self):
        """Adds thermal relaxation noise to model energy decay."""
        gate_time_ns = 50  # Gate time in nanoseconds
        error_thermal = thermal_relaxation_error(self.thermal_t1, self.thermal_t2, gate_time_ns)
        self.noise_model.add_all_qubit_quantum_error(error_thermal, ['x', 'h', 't'])

    def apply_readout_error(self):
        """Adds readout errors, which cause incorrect measurement results."""
        error_matrix = [[1 - self.readout_error_prob, self.readout_error_prob],  
                        [self.readout_error_prob, 1 - self.readout_error_prob]]  
        readout_error = ReadoutError(error_matrix)
        self.noise_model.add_all_qubit_readout_error(readout_error, [0, 1])

    def create_circuit(self):
        """Creates a quantum circuit with noise-prone gates."""
        circuit = QuantumCircuit(2, 2)

        circuit.h(0)       # Hadamard gate (creates superposition)
        circuit.t(1)       # T gate (introduces phase shift)
        circuit.cx(0, 1)   # CNOT gate (entanglement)
        circuit.measure([0, 1], [0, 1])  # Measurement

        return circuit

    def run_simulation(self, circuit):
        """Runs quantum circuit simulation with and without noise."""
        backend = AerSimulator()

        # Simulate ideal execution
        ideal_circuit = transpile(circuit, backend)
        ideal_result = execute(ideal_circuit, backend, shots=1000).result()
        ideal_counts = ideal_result.get_counts()

        # Simulate noisy execution
        noisy_circuit = transpile(circuit, backend, noise_model=self.noise_model)
        noisy_result = execute(noisy_circuit, backend, shots=1000).result()
        noisy_counts = noisy_result.get_counts()

        return ideal_counts, noisy_counts

    def monte_carlo_simulation(self, circuit, runs=10):
        """Runs Monte Carlo simulation to analyze noise effects over multiple runs."""
        backend = AerSimulator()

        fidelity_scores = []

        for _ in range(runs):
            noisy_circuit = transpile(circuit, backend, noise_model=self.noise_model)
            noisy_result = execute(noisy_circuit, backend, shots=1000).result()
            noisy_statevector = noisy_result.get_statevector()

            ideal_circuit = transpile(circuit, backend)
            ideal_result = execute(ideal_circuit, backend, shots=1000).result()
            ideal_statevector = ideal_result.get_statevector()

            fidelity = state_fidelity(ideal_statevector, noisy_statevector)
            fidelity_scores.append(fidelity)

        avg_fidelity = np.mean(fidelity_scores)
        print(f"\n📉 **Monte Carlo Fidelity Analysis**: Avg Fidelity over {runs} runs = {avg_fidelity:.5f}")

    def visualize_results(self, ideal_counts, noisy_counts):
        """Visualizes ideal vs. noisy results as histograms."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        plot_histogram(ideal_counts, title="✅ Ideal Quantum Circuit", ax=axes[0])
        plot_histogram(noisy_counts, title="⚠️ Noisy Quantum Circuit", ax=axes[1])

        plt.show()

if __name__ == "__main__":
    simulator = QuantumGateNoise()

    # Create Circuit
    circuit = simulator.create_circuit()
    print("\n🔹 Quantum Circuit:")
    print(circuit)

    # Run Simulations
    ideal_counts, noisy_counts = simulator.run_simulation(circuit)

    # Monte Carlo Simulation for Noise Analysis
    simulator.monte_carlo_simulation(circuit, runs=20)

    # Visualize Noisy vs. Ideal Results
    simulator.visualize_results(ideal_counts, noisy_counts)