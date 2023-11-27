# script to simulate the ising model
import numpy as np
import argparse
import os
# from simulator import Simulator
from para_sim import ParaSimulator


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=100)
    parser.add_argument('-T', type=float, default=1.)
    parser.add_argument('--num_sweeps', type=int, default=1e4)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('-J', type=int, default=1)
    parser.add_argument('--cold_start', type=bool, default=False)

    args = parser.parse_args()

    num_sweeps = int(args.num_sweeps)
    # remove num_sweeps from args
    del args.num_sweeps

    # simulator = Simulator(**vars(args))
    simulator = ParaSimulator(**vars(args))

    mags, energies, lattices = simulator.run(num_sweeps)

    data = {
        "magnetizations": mags,
        "energies": energies,
        "lattices": lattices,
    }
    data_dir = f"./data/T{args.T}/"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"seed{args.random_seed}.npz")
    np.savez(file_path, **data)
