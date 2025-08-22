import numpy as np
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Kraus
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError
import matplotlib.pyplot as plt
import seaborn as sns

#The Simulation Function
def simulation(Hx, Hz, decoder_fn, p, max_iter, N_trials=10E4):
    logical_failures = 0

    for _ in range(N_trials):
        initial_bitstring = "0" * Hx.shape[1]

        noisy_x_str = custom_noise_channel(p, initial_bitstring) #Pass through depolarizing noise
        noisy_z_str = custom_noise_channel(p, initial_bitstring)

        e_x = np.array([int(bit) for bit in noisy_x_str])
        e_z = np.array([int(bit) for bit in noisy_z_str])

        s_x = (Hx @ e_x) % 2
        s_z = (Hz @ e_z) % 2

        c_hat_x = decoder_fn(e=0.1, dec_max_iter = max_iter, bitstring=e_x, H=Hx) #Decode errors
        c_hat_z = decoder_fn(e=0.1, dec_max_iter = max_iter, bitstring=e_z, H=Hz)

        res_x = (e_x + c_hat_x) % 2  #Residual errors
        res_z = (e_z + c_hat_z) % 2

        valid_x = np.all((Hx @ res_x) % 2 == 0)
        valid_z = np.all((Hz @ res_z) % 2 == 0)

        if not (valid_x and valid_z):
            logical_failures += 1
        if logical_failures == 20:
            break
    return logical_failures / N_trials

#The decoder function
def decoder(e, dec_max_iter, bitstring, H):
    eps = 1e-10
    e = max(min(e, 1 - eps), eps)

    H = np.array(H)

    R = []     #Calculate initial LLRs (R)
    for i in bitstring:
        i = int(i)
        if i == 1:
            r = np.log(e / (1 - e))
        elif i == 0:
            r = np.log((1 - e) / e)
        else:
            raise ValueError("Bitstring must contain only 0s and 1s")
        R.append(r)

    R = np.array(R)

    M = np.zeros_like(H, dtype=float) #Initialize messages
    E = np.zeros_like(H, dtype=float)

    for _ in range(dec_max_iter):
        for i in range(H.shape[0]):  #Variable-to-check messages
            for j in range(H.shape[1]):
                if H[i, j] == 1:
                    M[i, j] = R[j]

        for i in range(H.shape[0]):#Check-to-variable messages
            for j in range(H.shape[1]):
                if H[i, j] == 1:
                    prod = 1.0
                    for k in range(H.shape[0]):
                        if H[k, j] == 1 and k != i:
                            prod *= np.tanh(M[k, j]/2)

                    numerator = max(1 + prod, eps)
                    denominator = max(1 - prod, eps)
                    E[i, j] = np.log(numerator / denominator)

        L = np.zeros(H.shape[1])#Belief update and hard decision
        C = np.zeros(H.shape[1])

        for j in range(H.shape[1]):
            L[j] = R[j]
            for i in range(H.shape[0]):
                if H[i, j] == 1:
                    L[j] += E[i, j]
            C[j] = 0 if L[j] >= 0 else 1

        # Step 5: Check if all parity checks are satisfied
        if np.all((H @ C) % 2 == 0):
            break

    return C

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

def plot_ler_vs_per(Hx, Hz, decoder_fn, p_range, N_trials=1000, target_failures=20, dec_max_iter_list=None):
    if dec_max_iter_list is None:
        dec_max_iter_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    rows = len(p_range)
    columns = len(dec_max_iter_list)

    logical_error_rates = np.zeros((rows, columns))

    for j, p in enumerate(p_range):
        print(f"\nPER = {p:.4f}")
        found = False
        for i, dec_max_iter in enumerate(dec_max_iter_list):
            failures = simulation(Hx, Hz, decoder_fn, p, dec_max_iter, N_trials)* N_trials
            print(f"  dec_max_iter={dec_max_iter}: logical failures={failures}")
            if failures >= target_failures:
                failures += 1 #DECODER FAILURES
            logical_error_rates[j, i] = failures / N_trials
    plt.figure(figsize=(10,6))
    for i, dec_max_iter in enumerate(dec_max_iter_list):
        plt.plot(p_range, logical_error_rates[:, i], label=f"Decoder Max Iter: {dec_max_iter}")
    plt.xlabel("Physical Error Rate (P)")
    plt.ylabel("Logical Error Rate (LER)")
    plt.title(f"LER vs PER (N_trials={N_trials})")
    plt.grid(True)
    plt.legend()
    plt.show()

#Example
#Here the HX and HZ are collected from the hypergraph product code of the 3- bit repetition code
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

plot_ler_vs_per(HX, HZ, decoder_fn=decoder, p_range=np.linspace(0.001, 0.005, 5), N_trials=10000, target_failures=20, dec_max_iter_list=[2, 5, 10])

