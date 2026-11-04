kem_keypair(alg) / kem_encap(alg, pk) / kem_decap(alg, sk, ct)     # ML-KEM
sig_keypair(alg) / sign(alg, sk, msg) / verify(alg, pk, msg, sig)   # ML-DSA (verify True/False)
x25519_keypair() / x25519_shared(sk, peer_pk)
hkdf(hash, ikm, salt, info, length) / digest(hash, data)
aead_seal(key, data, aad) / aead_open(key, blob, aad)               # AES-256-GCM
