# Summer Internship - 2025
# Quantum Error Correcting Codes
**Debashis Saikia**

*School of Physics, IISER, Thiruvananthapuram*

Under the guidance of 

**Dr. Arun B. Aloshious**

*Department of Electrical and Electronics Engineering, IIT, Guwahati*

---

This repository contains implementations and simulations of **quantum error correcting codes**, with a focus on **quantum low-density parity-check (qLDPC) codes** and **belief propagation–based decoding algorithms**.  
The project was developed during a research internship and aims to connect theoretical constructions with practical, simulation-ready code.

---

## Overview

Quantum computers are highly sensitive to noise, making quantum error correction essential for reliable computation. qLDPC codes are a promising class of quantum codes because they combine sparse stabilizers with favorable scaling properties, making them suitable for large-scale, fault-tolerant quantum architectures.

This repository explores:
- Construction of modern qLDPC codes
- Graph-based representations using Tanner graphs
- Message-passing decoders adapted to quantum codes

---

## Code Constructions

- **CSS Stabilizer Codes**  
  Implementations follow the CSS framework with separate X and Z parity checks.

- **Hypergraph Product Codes**  
  Constructed from two classical parity-check matrices, achieving sparse stabilizers, constant rate, and square-root distance scaling.

- **Lifted Product Codes**  
  A generalization of hypergraph product codes using permutation lifts, allowing improved distance scaling while maintaining sparsity.

---

## Decoding Algorithms

The following decoders are implemented and tested on quantum LDPC codes:

- **Belief Propagation (BP)**
- **Sum-Product Algorithm**
- **Min-Sum Approximation**
- **BP with Ordered Statistics Decoding (BP+OSD)**

These decoders operate on Tanner graphs and are adapted to handle quantum-specific features such as degeneracy.

---
