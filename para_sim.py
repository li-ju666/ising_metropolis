# Metropolis Monte Carlo simulation of the 2D Ising model
# Parallelized version with checkerboard pattern
import numpy as np
from dataclasses import dataclass


@dataclass
class ParaSimulator:
    N: int
    T: float
    random_seed: int
    lattice: np.ndarray = None
    cold_start: bool = False
    J: int = 1

    def __post_init__(self):
        # set the random seed
        np.random.seed(self.random_seed)

        # initialize the lattice
        if self.lattice is None:
            self.lattice = self.init_lattice(self.N, self.cold_start)

        # initialize checkerboard patterns
        self.black_mask = np.zeros((self.N, self.N), dtype=np.int8)
        self.black_mask[::2, ::2] = 1
        self.black_mask[1::2, 1::2] = 1
        self.black_mask = self.black_mask.astype(bool)

        self.white_mask = np.ones((self.N, self.N), dtype=np.int8)
        self.white_mask[::2, ::2] = 0
        self.white_mask[1::2, 1::2] = 0
        self.white_mask = self.white_mask.astype(bool)

        # precompute exponentials to speed up the simulation
        self.de_prob_dict = {
            delta_energy: np.exp(-float(delta_energy) / self.T)
            for delta_energy in range(-8, 9, 4)
        }

    def delta_energies_to_probs(self, delta_energies):
        vec_func = np.vectorize(
            lambda delta_energy: self.de_prob_dict[delta_energy])
        return vec_func(delta_energies)

    def mc_step(self, mask):
        # get neighbours
        up = np.roll(self.lattice, 1, axis=0)[mask]
        down = np.roll(self.lattice, -1, axis=0)[mask]
        left = np.roll(self.lattice, 1, axis=1)[mask]
        right = np.roll(self.lattice, -1, axis=1)[mask]

        # compute energy changes
        delta_energies = 2*self.J * self.lattice[mask] * (up+down+left+right)

        # compute the probabilities of flipping masked spins
        probs = self.delta_energies_to_probs(delta_energies)

        # compute masks for spins to be flipped
        flip_mask = np.random.rand(*probs.shape) < probs

        # merge the masks
        mask = mask.copy().reshape(-1)
        mask[mask] = flip_mask
        mask = mask.reshape(self.N, self.N)

        # flip the spins
        self.lattice[mask] *= -1

    def run(self, num_sweeps: int, verbose: bool = True):
        # initialize the arrays to store the magnetization and lattice
        magnetization_array = np.zeros(num_sweeps)
        energy_array = np.zeros(num_sweeps, dtype=np.int32)
        lattice_array = np.zeros((num_sweeps, self.N, self.N), dtype=np.int8)

        # run the simulation
        for current_sweep in range(num_sweeps):
            # step the lattice
            self.mc_step(self.black_mask)
            self.mc_step(self.white_mask)

            # compute the energy and magnetization
            energy = self.compute_energy(self.lattice, self.J)
            magnetization = self.lattice.mean()

            magnetization_array[current_sweep] = magnetization
            energy_array[current_sweep] = energy
            lattice_array[current_sweep] = self.lattice.astype(np.int8)

            if verbose and current_sweep % 100 == 0:
                print(f"Sweep {current_sweep}: E={energy}, M={magnetization}",
                      flush=True)
        return magnetization_array, energy_array, lattice_array

    # function to calculate the energy of the lattice
    # utilize periodic boundary conditions
    @staticmethod
    def compute_energy(lattice, J):
        up = np.roll(lattice, 1, axis=0)
        down = np.roll(lattice, -1, axis=0)
        left = np.roll(lattice, 1, axis=1)
        right = np.roll(lattice, -1, axis=1)

        energy = -J/2 * np.sum(lattice * (up+down+left+right))
        return energy

    # initialize the lattice
    @staticmethod
    def init_lattice(N, cold_start: bool = False):
        if cold_start:
            lattice = np.ones((N, N))
        else:
            lattice = np.random.choice(
                [-1, 1], size=(N, N))
        return lattice
