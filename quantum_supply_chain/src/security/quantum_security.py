import time
import numpy as np
from pqcrypto.kem.kyber1024 import generate_keypair, encrypt, decrypt
from pqcrypto.sign.dilithium5 import generate_keypair as sign_keypair, sign, verify
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

class QuantumSecurity:
    """
    Implements Post-Quantum Cryptography (PQC) for securing blockchain-based supply chains.
    Uses Kyber for Key Exchange & Dilithium for Digital Signatures.
    """

    def __init__(self):
        self.kyber_pub, self.kyber_priv = generate_keypair()  # Kyber PQC keypair
        self.dilithium_pub, self.dilithium_priv = sign_keypair()  # Dilithium PQC keypair

    def pqc_encrypt_decrypt(self, message: bytes):
        """
        Encrypts & decrypts using Kyber (Quantum-Resistant Key Encapsulation Mechanism).
        """
        ciphertext, shared_secret_enc = encrypt(self.kyber_pub)  # Encrypt
        decrypted_secret = decrypt(self.kyber_priv, ciphertext)  # Decrypt

        assert decrypted_secret == shared_secret_enc, "Decryption Failed!"
        return shared_secret_enc

    def pqc_sign_verify(self, message: bytes):
        """
        Signs & verifies a message using Dilithium (Quantum-Resistant Signatures).
        """
        signature = sign(self.dilithium_priv, message)
        is_valid = verify(self.dilithium_pub, message, signature)
        return is_valid

    def benchmark_classical_vs_pqc(self):
        """
        Benchmarks classical RSA vs. PQC algorithms for key generation, encryption, and signing.
        """
        rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        rsa_public_key = rsa_private_key.public_key()

        message = b"Secure blockchain transaction"

        # Benchmark RSA Encryption
        start_time = time.time()
        rsa_encrypted = rsa_public_key.encrypt(
            message,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
        )
        rsa_encrypt_time = time.time() - start_time

        # Benchmark RSA Signing
        start_time = time.time()
        rsa_signature = rsa_private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        rsa_sign_time = time.time() - start_time

        # Benchmark Kyber Encryption
        start_time = time.time()
        pqc_secret = self.pqc_encrypt_decrypt(message)
        pqc_encrypt_time = time.time() - start_time

        # Benchmark Dilithium Signing
        start_time = time.time()
        pqc_valid = self.pqc_sign_verify(message)
        pqc_sign_time = time.time() - start_time

        print("\n📊 **Benchmark Results (RSA vs. PQC):**")
        print(f"🔒 RSA Encryption Time: {rsa_encrypt_time:.5f} sec")
        print(f"🔑 RSA Signing Time: {rsa_sign_time:.5f} sec")
        print(f"🚀 Kyber Encryption Time: {pqc_encrypt_time:.5f} sec")
        print(f"✅ Dilithium Signing Time: {pqc_sign_time:.5f} sec")

if __name__ == "__main__":
    security = QuantumSecurity()
    security.benchmark_classical_vs_pqc()