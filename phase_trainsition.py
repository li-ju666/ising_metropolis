# script to simulate the ising model
import numpy as np
import os
from simulator import Simulator
from para_sim import ParaSimulator


if __name__ == '__main__':
    N = 100
    J = 1
    num_sweeps = int(1e4)

    T1 = 1.0
    T2 = 2.0
    T3 = 2.5
    T4 = 3.5

    Ts1 = np.linspace(T1, T2, 10)
    Ts2 = np.linspace(T2, T3, 30)
    Ts3 = np.linspace(T3, T4, 10)

    Ts = np.concatenate([Ts1, Ts2, Ts3])

    lattice = np.random.choice([-1, 1], size=(N, N))

    for T_idx, T in enumerate(Ts):
        # simulator = Simulator(
        simulator = ParaSimulator(
            N=100, T=T, random_seed=42, lattice=lattice, J=J,
        )

        mags, energies, lattices = simulator.run(num_sweeps)

        data = {
            "magnetizations": mags,
            "energies": energies,
            "lattices": lattices,
        }
        data_dir = "./data/crit_temp/"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"{T_idx}.npz")
        np.savez(file_path, **data)

        # reuse the last lattice as the initial lattice
        lattice = lattices[-1]
