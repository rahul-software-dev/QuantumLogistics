import unittest
from src.quantum_processing.quantum_error_correction import QuantumErrorMitigator

class TestQuantumErrorMitigator(unittest.TestCase):
    def test_calibrate_and_mitigate(self):
        mitigator = QuantumErrorMitigator()
        mitigator.calibrate(num_qubits=2, shots=10)
        # Simulate raw counts
        raw_counts = {'00': 50, '01': 30, '10': 15, '11': 5}
        mitigated = mitigator.mitigate(raw_counts)
        self.assertIsInstance(mitigated, dict)

if __name__ == '__main__':
    unittest.main()
