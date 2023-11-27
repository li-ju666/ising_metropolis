# Metropolis Monte Carlo simulation of the 2D Ising model
import numpy as np
from dataclasses import dataclass


@dataclass
class Simulator:
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

        # precompute exponentials to speed up the simulation
        self.delta_energy_prob_dict = {
            delta_energy: np.exp(-float(delta_energy) / self.T)
            for delta_energy in range(-8, 9, 4)
        }

    # compute the change in energy if the spin at (i, j) is flipped
    def compute_delta_energy(self, i, j):
        N = self.N
        up = self.lattice[(i - 1) % N, j]
        down = self.lattice[(i + 1) % N, j]
        left = self.lattice[i, (j - 1) % N]
        right = self.lattice[i, (j + 1) % N]

        delta_energy = 2*self.J * self.lattice[i, j] * (up+down+left+right)
        return delta_energy

    # compute the probability of flipping the spin
    def compute_flip_prob(self, delta_energy):
        return self.delta_energy_prob_dict[delta_energy]

    def mc_step(self, i, j):
        # compute the change in energy
        delta_energy = self.compute_delta_energy(i, j)

        # compute the probability of flipping the spin
        flip_prob = self.compute_flip_prob(delta_energy)

        # flip the spin with probability flip_prob
        if np.random.rand() < flip_prob:
            self.lattice[i, j] *= -1
            delta_mag_sum = 2*self.lattice[i, j]
        else:
            delta_energy = 0
            delta_mag_sum = 0

        return delta_energy, delta_mag_sum

    def run(self, num_sweeps: int, verbose: bool = True):
        n_iter = num_sweeps * self.N**2

        # initialize the arrays to store the magnetization and lattice
        magnetization_array = np.zeros(num_sweeps)
        energy_array = np.zeros(num_sweeps, dtype=np.int32)
        lattice_array = np.zeros((num_sweeps, self.N, self.N), dtype=np.int8)

        # generate random indices
        ijs = np.random.randint(0, self.N, size=(n_iter, 2))

        # initialize the metrics
        energy = self.compute_energy(self.lattice, self.J)
        mag_sum = self.lattice.sum()

        # run the simulation
        for iter_i, (i, j) in enumerate(ijs):
            # step the lattice
            delta_energy, delta_mag_sum = self.mc_step(i, j)

            # update the metrics
            energy += delta_energy
            mag_sum += delta_mag_sum

            # save the lattice every sweep
            if iter_i % (self.N**2) == 0:
                current_sweep = iter_i // (self.N**2)
                magnetization_array[current_sweep] = (mag_sum / self.N**2)
                energy_array[current_sweep] = energy
                lattice_array[current_sweep] = self.lattice.astype(np.int8)

                if verbose and current_sweep % 100 == 0:
                    print(f"Sweep {current_sweep}: E={energy}, " +
                          f"M={mag_sum / self.N**2}", flush=True)
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
