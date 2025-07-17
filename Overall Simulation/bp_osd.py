pip install -U ldpc

import ldpc
print(ldpc.__version__)

pip install qiskit

pip install qiskit-aer

"""###Importing Libraries"""

import numpy as np
import ldpc.codes
from ldpc import BpOsdDecoder
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Kraus
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError
import matplotlib.pyplot as plt
import seaborn as sns

"""###Noise layer"""

def custom_noise_channel(p_error, initial_bitstring):
    n_qubits = len(initial_bitstring)
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Set the initial state from bitstring
    for i, bit in enumerate(initial_bitstring):
        if bit == '1':
            qc.x(i)

    # Define Kraus operators for full depolarizing channel
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    k0 = np.sqrt(1 - p_error) * I
    k1 = np.sqrt(p_error / 3) * X
    k2 = np.sqrt(p_error / 3) * Y
    k3 = np.sqrt(p_error / 3) * Z

    depolarizing_channel = Kraus([k0, k1, k2, k3])

    # Define noise model
    noise_model = NoiseModel()
    for qubit in range(n_qubits):
        noise_model.add_quantum_error(depolarizing_channel, ['id'], [qubit])
        qc.id(qubit)  # Apply identity so the noise attaches here

    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))

    # Simulate
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc, shots=1).result()
    counts = result.get_counts()
    noisy_bitstring = list(counts.keys())[0]

    return noisy_bitstring

"""###Simulation Function"""

def simulation(Hx, Hz, decoder, p, N_trials):
    p = float(p)

    # instantiate decoders for this p
    bpz = decoder(pcm = Hx, error_rate=p, max_iter=20, bp_method="product_sum", schedule = 'serial', osd_method="osd_0", osd_order=0)
    bpx = decoder(pcm = Hz, error_rate=p, max_iter=20, bp_method="product_sum", schedule = 'serial', osd_method="osd_0", osd_order=0)

    logical_failures = 0

    for _ in range(N_trials):
        noisy_x_str = custom_noise_channel(p, "0" * Hx.shape[1])
        noisy_z_str = custom_noise_channel(p, "0" * Hz.shape[1])

        e_x = np.array([int(bit) for bit in noisy_x_str])
        e_z = np.array([int(bit) for bit in noisy_z_str])

        s_x = (Hx @ e_z).A1 % 2  # convert matrix->array
        s_z = (Hz @ e_x).A1 % 2

        c_hat_z = bpz.decode(s_x)
        c_hat_x = bpx.decode(s_z)

        res_x = (e_x + c_hat_x) % 2
        res_z = (e_z + c_hat_z) % 2

        valid_x = np.all((Hx @ res_z).A1 % 2 == 0)
        valid_z = np.all((Hz @ res_x).A1 % 2 == 0)

        if not (valid_x and valid_z):
            logical_failures += 1

    return logical_failures / N_trials

"""###Plotting function"""

def plot_logical_error_rate(Hx, Hz, decoder, N_trials=500, p_range=None, title="Logical Error Rate vs Physical Error Rate"):
    if p_range is None:
        p_range = np.linspace(0.01, 0.2, 10)

    logical_error_rates = []

    for p in p_range:
        print(f"Running simulation at p = {p:.3f}")
        ler = simulation(Hx, Hz, decoder, p, N_trials)  # calls your bposd-backed simulation
        logical_error_rates.append(ler)

    # Plotting with seaborn
    sns.set(style="whitegrid", context="talk", font_scale=0.5)
    palette = sns.color_palette("deep")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=p_range, y=logical_error_rates, marker='o', linewidth=2.5, color=palette[0])
    plt.title(title)
    plt.xlabel("Physical Error Rate (p)")
    plt.ylabel("Logical Error Rate (LER)")
    plt.yscale("log")  # shows exponential decay if it exists
    plt.xticks(np.round(p_range, 3))
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return logical_error_rates

"""###Parity check matrices"""

#Example
HX = np.matrix([[1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1]])
HZ = np.matrix([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
 [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]])

"""###Decoder"""

plot_logical_error_rate(Hx=HX, Hz=HZ, decoder=BpOsdDecoder, N_trials=300, p_range=np.linspace(0.01, 1, 20))

