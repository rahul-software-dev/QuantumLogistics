import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation, Unroller, Optimize1qGates, CXCancellation
from qiskit.providers.aer.noise import NoiseModel

class QuantumCircuitOptimizer:
    """
    Research-Grade Quantum Circuit Optimizer.
    Optimizes circuits for execution on real quantum hardware by:
    - Reducing circuit depth and gate count
    - Applying noise-aware transpilation
    - Using custom transpilation passes
    - Visualizing gate operations before and after optimization
    """

    def __init__(self, backend_name="qasm_simulator"):
        self.backend = Aer.get_backend(backend_name)
        self.noise_model = None  # Optional: Set a noise model for realistic optimization

    def set_noise_model(self, noise_model: NoiseModel):
        """Allows setting a noise model for hardware-aware optimization."""
        self.noise_model = noise_model

    def optimize(self, circuit: QuantumCircuit, optimization_level=3):
        """
        Uses Qiskit's built-in transpiler for circuit optimization.
        
        :param circuit: QuantumCircuit object
        :param optimization_level: Optimization level (0-3)
        :return: Optimized QuantumCircuit
        """
        optimized_circuit = transpile(circuit, backend=self.backend, optimization_level=optimization_level)
        return optimized_circuit

    def apply_custom_passes(self, circuit: QuantumCircuit):
        """
        Uses custom transpilation passes for more advanced optimizations.
        
        :param circuit: QuantumCircuit object
        :return: Optimized QuantumCircuit
        """
        pass_manager = PassManager([
            Unroller(['u3', 'cx']),        # Convert to standard gates
            Optimize1qGates(),             # Simplify 1-qubit gates
            CommutativeCancellation(),     # Remove redundant operations
            CXCancellation()               # Optimize CNOT gates
        ])
        optimized_circuit = pass_manager.run(circuit)
        return optimized_circuit

    def hardware_aware_optimization(self, circuit: QuantumCircuit, backend):
        """
        Uses a noise-aware transpilation method to optimize for a given quantum device.
        
        :param circuit: QuantumCircuit object
        :param backend: Quantum backend for transpilation (e.g., IBMQ device)
        :return: Optimized QuantumCircuit
        """
        optimized_circuit = transpile(circuit, backend=backend, optimization_level=3, noise_model=self.noise_model)
        return optimized_circuit

    def benchmark(self, original: QuantumCircuit, optimized: QuantumCircuit):
        """
        Compares the original and optimized circuits in terms of depth and gate count.
        
        :param original: Original QuantumCircuit
        :param optimized: Optimized QuantumCircuit
        """
        print("\n🔍 **Optimization Benchmarking**")
        print(f"⚡ Original Depth: {original.depth()} → Optimized Depth: {optimized.depth()}")
        print(f"⚡ Original Gate Count: {original.count_ops()} → Optimized Gate Count: {optimized.count_ops()}")

        # Plot optimization comparison
        self.plot_gate_usage(original, optimized)

    def plot_gate_usage(self, original: QuantumCircuit, optimized: QuantumCircuit):
        """
        Plots a comparative analysis of gate usage before and after optimization.
        """
        original_gates = original.count_ops()
        optimized_gates = optimized.count_ops()

        gate_types = set(original_gates.keys()).union(set(optimized_gates.keys()))
        original_counts = [original_gates.get(g, 0) for g in gate_types]
        optimized_counts = [optimized_gates.get(g, 0) for g in gate_types]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(gate_types))
        ax.bar(x - 0.2, original_counts, width=0.4, label="Original", color="red")
        ax.bar(x + 0.2, optimized_counts, width=0.4, label="Optimized", color="green")

        ax.set_xticks(x)
        ax.set_xticklabels(gate_types)
        ax.set_ylabel("Gate Count")
        ax.set_title("Gate Usage Before and After Optimization")
        ax.legend()
        plt.show()

    def visualize_circuit(self, circuit: QuantumCircuit):
        """
        Uses NetworkX to create a graph representation of the quantum circuit.
        
        :param circuit: QuantumCircuit object
        """
        graph = nx.DiGraph()

        for i, instruction in enumerate(circuit.data):
            gate, qubits, _ = instruction
            for qubit in qubits:
                graph.add_edge(i, qubit.index, label=gate.name)

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000)
        labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        plt.title("Quantum Circuit Graph Representation")
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Sample quantum circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.h(1)
    qc.cx(0, 2)

    print("🔹 Original Circuit:")
    print(qc)

    optimizer = QuantumCircuitOptimizer()
    optimized_qc = optimizer.optimize(qc, optimization_level=3)

    print("\n🔹 Optimized Circuit:")
    print(optimized_qc)

    optimizer.benchmark(qc, optimized_qc)
    optimizer.visualize_circuit(optimized_qc)