"""Phase 0 PQC smoke test: ML-KEM-768 key exchange and ML-DSA-65 signatures (liboqs)."""
import time

import oqs


def ms(t0):
    return (time.perf_counter() - t0) * 1000


def kem_test():
    # The server publishes a public key; the client uses it to create a shared secret.
    with oqs.KeyEncapsulation("ML-KEM-768") as server:
        t = time.perf_counter()
        public_key = server.generate_keypair()
        t_keygen = ms(t)

        with oqs.KeyEncapsulation("ML-KEM-768") as client:
            t = time.perf_counter()
            ciphertext, secret_client = client.encap_secret(public_key)
            t_encap = ms(t)

        t = time.perf_counter()
        secret_server = server.decap_secret(ciphertext)
        t_decap = ms(t)

    assert secret_client == secret_server, "shared secrets differ!"
    print(f"ML-KEM-768  public key {len(public_key)} B | ciphertext {len(ciphertext)} B | "
          f"secret {len(secret_server)} B")
    print(f"            keygen {t_keygen:.2f} ms | encap {t_encap:.2f} ms | decap {t_decap:.2f} ms")


def sig_test():
    # The server signs a global model; clients check the signature.
    model_bytes = b"global model weights, round 1"
    with oqs.Signature("ML-DSA-65") as signer:
        public_key = signer.generate_keypair()
        t = time.perf_counter()
        signature = signer.sign(model_bytes)
        t_sign = ms(t)

    with oqs.Signature("ML-DSA-65") as verifier:
        t = time.perf_counter()
        ok = verifier.verify(model_bytes, signature, public_key)
        t_verify = ms(t)
        tampered = verifier.verify(b"tampered model weights", signature, public_key)

    assert ok, "valid signature rejected!"
    assert not tampered, "tampered model accepted!"
    print(f"ML-DSA-65   public key {len(public_key)} B | signature {len(signature)} B")
    print(f"            sign {t_sign:.2f} ms | verify {t_verify:.2f} ms | tampered model rejected")


if __name__ == "__main__":
    print("liboqs", oqs.oqs_version())
    kem_test()
    sig_test()
    print("PQ SMOKE TEST: PASS")