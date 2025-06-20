# Hypergraph Product Code

Hypergraph Product Codes are a powerful class of quantum error-correcting codes (QECCs) that help protect quantum information from errors due to noise and decoherence, a major challenge in building scalable quantum computers.

These codes are best defined in terms of hypergraphs (graphs in which one edge can join more than two vertices), but they can also be defined in terms of **matrices**. The matrix representation is the most frequently used in quantum error correction theory and implementation.

Hypergraph Product Codes were introduced by **Tillich and Zémor (2009)**. They’re a type of **Calderbank-Shor-Steane (CSS) code** constructed by taking two classical binary linear codes and combining them via a hypergraph product. This gives us a **low-density parity-check (LDPC) quantum code** — where each qubit and stabilizer involves only a few interactions. That makes HPCs ideal for fault-tolerant quantum computing.

---

## Code Definition and Matrix Construction

Let there be two classical binary codes:

- \[n<sub>a</sub>, k<sub>a</sub>, d<sub>a</sub>\] of C(A)
- \[n<sub>b</sub>, k<sub>b</sub>, d<sub>b</sub>\] of C(B)

Let the corresponding parity check matrices be **A** and **B**, respectively.

Then, the hypergraph product code is the CSS code defined by the stabilizer matrices **H<sub>X</sub>** and **H<sub>Z</sub>**, where:

### X-Stabilizer Matrix H<sub>X</sub>

```math
H_X = [
    A ⊗ I<sub>n<sub>B</sub></sub>  |  I<sub>m<sub>A</sub></sub> ⊗ B
]
```
### Z-Stabilizer Matrix H<sub>Z</sub>

```math
H_Z = [
    I<sub>n<sub>A</sub></sub>⊗B<sup>T</sup>∣A<sup>T</sup>⊗I<sub>m<sub>B</sub></sub>
]
```

Check out the original paper by **Tillich and Zémor ** [Quantum LDPC Codes With Positive Rate and Minimum Distance Proportional to the Square Root of the Blocklength](https://ieeexplore.ieee.org/document/6671468) for more information.
