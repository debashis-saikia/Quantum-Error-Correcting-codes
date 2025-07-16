## BP-SP: Sum-Product Decoding

The **Sum-Product Algorithm (SPA)** or BP decoding is a probabilistic message-passing method used on Tanner graphs to infer the most likely error given syndrome data. It works by iteratively updating messages between variable nodes (qubits) and check nodes (stabilizers), exchanging likelihoods until convergence. In the quantum context, SPA is typically adapted to handle degeneracy and dual (X/Z) error channels, though classical BP remains foundational. While efficient and scalable, SPA suffers on highly loopy graphs or for errors beyond its reach — which is where hybrid schemes like BP-OSD shine.

This decoder is heavily based on iterative decoding principles outlined in Sarah J. Johnson’s book *"Iterative Error Correction: Turbo, Low-Density Parity-Check and Repeat-Accumulate Codes"*, which provides a clear and detailed view of the sum-product algorithm and its practical applications.

### Reference
- Sarah J. Johnson, *Iterative Error Correction: Turbo, Low-Density Parity-Check and Repeat-Accumulate Codes*, Cambridge University Press, 2010.

---


## BP-OSD: Belief Propagation with Ordered Statistics Decoding

**BP-OSD** is a powerful hybrid decoding algorithm that first applies standard Belief Propagation to estimate soft information and then performs a refinement using Ordered Statistics Decoding (OSD). After running a fixed number of BP iterations (or until convergence), the decoder uses the reliabilities (e.g., log-likelihood ratios) to reframe the decoding problem as a syndrome decoding task, effectively guessing the most likely error pattern within a limited basis space. This two-step method drastically improves decoding in high-noise or loopy graph conditions where BP alone struggles.

OSD reorders bits by reliability, reduces the decoding problem to the most confident subspace, and uses Gaussian elimination to solve for corrections, making it one of the best known classical-inspired decoding tricks for quantum error correction.

### Reference
- P. Panteleev and G. Kalachev, "Degenerate Quantum LDPC Codes With Good Finite Length Performance," *Quantum*, vol. 5, p. 585, Nov. 2021. [Online]. Available: https://doi.org/10.22331/q-2021-11-22-585

---
