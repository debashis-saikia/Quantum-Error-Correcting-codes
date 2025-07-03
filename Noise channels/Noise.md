# Quantum Noise Channels 

In practical quantum computing, qubits are not perfectly isolated. Their interaction with the environment introduces errors known as **quantum noise**, leading to decoherence and loss of information.

Quantum noise is modeled mathematically using **quantum channels**, which describe how an initial quantum state evolves under such disturbances. These channels are represented as **Completely Positive Trace Preserving (CPTP)** maps to ensure physically valid operations.

## Operator-Sum /Kraus Representation

The evolution of a quantum state $\rho$ under a general noise process is given by the operator-sum or Kraus representation:

$$
\rho' = \sum_k K_k \, \rho \, K_k^\dagger
$$

Where:
- $\rho$ is the density matrix of the quantum state before noise.
- $K_k$ are the **Kraus operators**, satisfying the completeness condition:

$$
\sum_k K_k^\dagger K_k = I
$$

- $\rho'$ is the noisy output state.

Note that the Kraus operators generalize both unitary and non-unitary evolution. 

## Choi-Kraus-Sudarshan Theorem

The **Choi-Kraus-Sudarshan theorem** states that:

> Any completely positive (CP) linear map can be expressed using a set of Kraus operators, and every CPTP map corresponds to a physically valid quantum operation on density matrices.

## Common Noise Channels

Examples of noise modeled via Kraus operators include:

- **Bit Flip Channel** — applies $X$ (Pauli-X) with probability $p$
- **Phase Flip Channel** — applies $Z$ (Pauli-Z) with probability $p$
- **Bit-Phase Flip Channel** — applies $Y$ (Pauli-Y) with probability $p$
- **Depolarizing Channel** — randomly applies $X$, $Y$, or $Z$ errors isotropically

## References

1. Kraus, K. *States, Effects, and Operations: Fundamental Notions of Quantum Theory*, Springer, 1983.  
2. Choi, M. D. *Completely Positive Linear Maps on Complex Matrices*, Linear Algebra and Its Applications, 10(3), 285–290 (1975).  
3. Sudarshan, E. C. G., Mathews, P. M., & Rau, J. *Stochastic Dynamics of Quantum-Mechanical Systems*, Physical Review, 121(3), 920–924 (1961).  

These references form the mathematical backbone for understanding quantum noise.
