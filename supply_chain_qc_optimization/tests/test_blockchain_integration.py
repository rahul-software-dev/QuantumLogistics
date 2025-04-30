import unittest
from src.blockchain_integration.blockchain_integration import Blockchain

class TestBlockchain(unittest.TestCase):
    def setUp(self):
        self.blockchain = Blockchain()

    def test_add_block_and_validate(self):
        self.blockchain.add_block({'data': 'test'})
        self.assertTrue(self.blockchain.is_chain_valid())

if __name__ == '__main__':
    unittest.main()
