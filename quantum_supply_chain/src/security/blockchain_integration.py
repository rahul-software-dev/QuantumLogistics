import hashlib
import json
import time
import pqcrypto.sign.dilithium3 as pq_dilithium
import pqcrypto.kem.kyber512 as pq_kyber
from typing import List, Dict
from flask import Flask, request

class Blockchain:
    """
    Private blockchain implementation for securing supply chain logistics.
    Uses quantum-resistant cryptography for digital signatures and key exchange.
    """

    def __init__(self):
        self.chain: List[Dict] = []
        self.pending_transactions: List[Dict] = []
        self.nodes: set = set()

        # Generate Genesis Block
        self.create_block(previous_hash="1", proof=100)

        # Quantum-Resistant Key Generation
        self.public_key, self.secret_key = pq_kyber.generate_keypair()

    def create_block(self, proof: int, previous_hash: str):
        """
        Creates a new block in the blockchain.
        """
        block = {
            "index": len(self.chain) + 1,
            "timestamp": time.time(),
            "transactions": self.pending_transactions,
            "proof": proof,
            "previous_hash": previous_hash or self.hash(self.chain[-1]),
        }
        self.pending_transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, sender: str, receiver: str, product_id: str, status: str):
        """
        Adds a transaction to the pending transactions list.
        """
        transaction = {
            "sender": sender,
            "receiver": receiver,
            "product_id": product_id,
            "status": status,
            "timestamp": time.time(),
        }

        # Sign the transaction with quantum-resistant Dilithium signature
        signature = self.sign_transaction(transaction)
        transaction["signature"] = signature

        self.pending_transactions.append(transaction)
        return self.last_block["index"] + 1

    def sign_transaction(self, transaction: Dict) -> str:
        """
        Uses Dilithium post-quantum signatures to sign transactions.
        """
        transaction_data = json.dumps(transaction, sort_keys=True).encode()
        signature = pq_dilithium.sign(transaction_data, self.secret_key)
        return signature.hex()

    def verify_transaction(self, transaction: Dict) -> bool:
        """
        Verifies the quantum-resistant signature.
        """
        transaction_data = json.dumps(transaction, sort_keys=True).encode()
        try:
            pq_dilithium.verify(transaction["signature"].encode(), transaction_data, self.public_key)
            return True
        except:
            return False

    @staticmethod
    def hash(block: Dict) -> str:
        """
        Creates a SHA-256 hash of a block.
        """
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof: int) -> int:
        """
        A basic Proof-of-Work (PoW) algorithm to add blocks securely.
        """
        new_proof = 1
        check_proof = False

        while not check_proof:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:5] == "00000":  # Adjust complexity as needed
                check_proof = True
            else:
                new_proof += 1

        return new_proof


# Flask API for Blockchain Interaction
app = Flask(__name__)
blockchain = Blockchain()

@app.route('/mine_block', methods=['GET'])
def mine_block():
    """
    Mines a new block and adds it to the blockchain.
    """
    previous_proof = blockchain.last_block["proof"]
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(blockchain.last_block)
    block = blockchain.create_block(proof, previous_hash)

    response = {
        "message": "New Block Mined",
        "index": block["index"],
        "transactions": block["transactions"],
        "proof": block["proof"],
        "previous_hash": block["previous_hash"],
    }
    return response, 200

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    """
    Adds a new supply chain transaction to the blockchain.
    """
    json_data = request.get_json()
    required_fields = ["sender", "receiver", "product_id", "status"]

    if not all(field in json_data for field in required_fields):
        return {"message": "Missing required fields"}, 400

    index = blockchain.add_transaction(
        sender=json_data["sender"],
        receiver=json_data["receiver"],
        product_id=json_data["product_id"],
        status=json_data["status"]
    )

    return {"message": f"Transaction added to block {index}"}, 201

@app.route('/get_chain', methods=['GET'])
def get_chain():
    """
    Returns the entire blockchain.
    """
    response = {"chain": blockchain.chain, "length": len(blockchain.chain)}
    return response, 200

@app.route('/is_valid', methods=['GET'])
def is_valid():
    """
    Checks the validity of the blockchain.
    """
    chain = blockchain.chain
    for i in range(1, len(chain)):
        block, prev_block = chain[i], chain[i - 1]
        if block["previous_hash"] != blockchain.hash(prev_block):
            return {"message": "Blockchain is invalid"}, 400
        if not blockchain.verify_transaction(block["transactions"][0]):  # Check first transaction
            return {"message": "Invalid transaction detected"}, 400

    return {"message": "Blockchain is valid"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)