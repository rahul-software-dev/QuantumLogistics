import time
import secrets
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
        print("🔐 Initializing Post-Quantum Cryptography Module...")
        try:
            # Generate Kyber PQC Keypair for Key Encapsulation (KEM)
            self.kyber_pub, self.kyber_priv = generate_keypair()

            # Generate Dilithium PQC Keypair for Digital Signatures
            self.dilithium_pub, self.dilithium_priv = sign_keypair()

            print("✅ PQC Key Pairs Successfully Generated!")
        except Exception as e:
            print(f"❌ Error Initializing PQC Keys: {e}")

    def pqc_encrypt_decrypt(self):
        """
        Encrypts & decrypts using Kyber (Quantum-Resistant Key Encapsulation Mechanism).
        :return: Decrypted shared secret (should match the encapsulated one)
        """
        try:
            # Encrypt a randomly generated secret
            ciphertext, shared_secret_enc = encrypt(self.kyber_pub)

            # Decrypt the ciphertext
            decrypted_secret = decrypt(self.kyber_priv, ciphertext)

            if decrypted_secret != shared_secret_enc:
                raise ValueError("Decryption failed! Shared secrets do not match.")

            print("🔑 Kyber Encryption & Decryption Successful!")
            return shared_secret_enc
        except Exception as e:
            print(f"❌ PQC Encryption/Decryption Error: {e}")

    def pqc_sign_verify(self, message: bytes):
        """
        Signs & verifies a message using Dilithium (Quantum-Resistant Signatures).
        :param message: The message to sign.
        :return: Boolean indicating if the signature is valid.
        """
        try:
            # Sign the message
            signature = sign(self.dilithium_priv, message)

            # Verify the signature
            is_valid = verify(self.dilithium_pub, message, signature)

            if is_valid:
                print("✅ Dilithium Signature Successfully Verified!")
            else:
                print("❌ Dilithium Signature Verification Failed!")

            return is_valid
        except Exception as e:
            print(f"❌ PQC Signing/Verification Error: {e}")

    def benchmark_classical_vs_pqc(self):
        """
        Benchmarks classical RSA vs. PQC algorithms for key generation, encryption, and signing.
        """
        print("\n⏳ Running Benchmark... Please wait.")

        # Generate a 2048-bit RSA Keypair
        try:
            rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            rsa_public_key = rsa_private_key.public_key()
        except Exception as e:
            print(f"❌ RSA Key Generation Error: {e}")
            return

        # Message for testing cryptographic operations
        message = secrets.token_bytes(32)  # 32-byte random message

        # --- RSA ENCRYPTION ---
        try:
            start_time = time.time()
            rsa_encrypted = rsa_public_key.encrypt(
                message,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
            )
            rsa_encrypt_time = time.time() - start_time
        except Exception as e:
            print(f"❌ RSA Encryption Error: {e}")
            return

        # --- RSA SIGNING ---
        try:
            start_time = time.time()
            rsa_signature = rsa_private_key.sign(
                message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            rsa_sign_time = time.time() - start_time
        except Exception as e:
            print(f"❌ RSA Signing Error: {e}")
            return

        # --- PQC KYBER ENCRYPTION ---
        start_time = time.time()
        pqc_secret = self.pqc_encrypt_decrypt()
        pqc_encrypt_time = time.time() - start_time

        # --- PQC DILITHIUM SIGNING ---
        start_time = time.time()
        pqc_valid = self.pqc_sign_verify(message)
        pqc_sign_time = time.time() - start_time

        # --- DISPLAY RESULTS ---
        print("\n📊 **Benchmark Results (RSA vs. PQC):**")
        print(f"🔒 RSA Encryption Time: {rsa_encrypt_time:.5f} sec")
        print(f"🔑 RSA Signing Time: {rsa_sign_time:.5f} sec")
        print(f"🚀 Kyber Encryption Time: {pqc_encrypt_time:.5f} sec")
        print(f"✅ Dilithium Signing Time: {pqc_sign_time:.5f} sec")

        # --- VERIFICATION ---
        print("\n🔎 **Verification Summary:**")
        if pqc_valid:
            print("✅ PQC Signature Verified Successfully!")
        else:
            print("❌ PQC Signature Verification Failed!")

    def save_keys(self):
        """
        Saves the PQC key pairs to files securely.
        """
        try:
            with open("kyber_pub.key", "wb") as f:
                f.write(self.kyber_pub)

            with open("kyber_priv.key", "wb") as f:
                f.write(self.kyber_priv)

            with open("dilithium_pub.key", "wb") as f:
                f.write(self.dilithium_pub)

            with open("dilithium_priv.key", "wb") as f:
                f.write(self.dilithium_priv)

            print("✅ PQC Keys Successfully Saved!")
        except Exception as e:
            print(f"❌ Error Saving Keys: {e}")

    def load_keys(self):
        """
        Loads PQC key pairs from files.
        """
        try:
            with open("kyber_pub.key", "rb") as f:
                self.kyber_pub = f.read()

            with open("kyber_priv.key", "rb") as f:
                self.kyber_priv = f.read()

            with open("dilithium_pub.key", "rb") as f:
                self.dilithium_pub = f.read()

            with open("dilithium_priv.key", "rb") as f:
                self.dilithium_priv = f.read()

            print("✅ PQC Keys Successfully Loaded!")
        except Exception as e:
            print(f"❌ Error Loading Keys: {e}")


# Example Usage
if __name__ == "__main__":
    security = QuantumSecurity()

    # Benchmark RSA vs. PQC Performance
    security.benchmark_classical_vs_pqc()

    # Save and Load PQC Keys
    security.save_keys()
    security.load_keys()