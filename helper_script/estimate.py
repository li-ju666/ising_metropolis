import numpy as np
from matplotlib import pyplot as plt
import os


def read(T, random_seed):
    data = np.load(f"./data/T{T}/seed{random_seed}.npz")
    magnetizations = data["magnetizations"]
    energies = data["energies"]
    return magnetizations, energies


def estimate(data, delta_t, tau):
    t = data.shape[0]
    hat_m = np.mean(data)
    hat_m2 = np.mean(data**2)
    error = np.sqrt(
        (hat_m2 - hat_m**2)*(1+2*tau/delta_t)/(t-1))
    return hat_m, error


if __name__ == "__main__":
    Ts = [1.0, 2.0, 4.0]
    seed = 0

    for T in Ts:
        mag, energy = read(T, seed)
        eq_idx = int(5e3)
        mag, energy = mag[eq_idx:], energy[eq_idx:]

        delta_t = 1e4
        t = mag.shape[0]
        tau = 1e7 if T == 2.0 else 5e6

        hat_m, error = estimate(mag, delta_t, tau)
        print(f"T = {T}, hat_m = {hat_m}, error = {error}")

        hat_e, error = estimate(energy, delta_t, tau)
        print(f"T = {T}, hat_e = {hat_e}, error = {error}")
        print("=======")
