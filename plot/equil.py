# script to plot a single simulation
import numpy as np
from matplotlib import pyplot as plt
import os


def read(T, random_seed):
    data = np.load(f"./data/T{T}/seed{random_seed}.npz")
    magnetizations = data["magnetizations"]
    energies = data["energies"]
    return magnetizations, energies


def plot_magnetization(magnetizations, T, name=None):
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for seed, mag in enumerate(magnetizations):
        ax.plot(np.abs(mag), alpha=0.8, label=f"seed {seed}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Magnetization")
    ax.set_title(f"Magnetization at $T = {T}$")
    # set limits
    ax.set_ylim(0, 1.1)
    ax.legend()
    if name:
        plt.savefig(name, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_energy(energies, T, name=None):
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for seed, energy in enumerate(energies):
        ax.plot(energy, alpha=0.8, label=f"seed {seed}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title(f"Energy at $T = {T}$")
    ax.legend()
    if name:
        plt.savefig(name, dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    T = 2.4
    seeds = [0, 1, 2]

    mags = []
    energies = []
    for seed in seeds:
        mag, energy = read(T, seed)
        mags.append(mag)
        energies.append(energy)

    # plot the magnetization
    figure_path = "./figures/"
    os.makedirs(figure_path, exist_ok=True)

    mag_figure_name = os.path.join(figure_path, f"mag_T{T}.pdf")
    plot_magnetization(mags, T, mag_figure_name)

    # plot the energy
    energy_figure_name = os.path.join(figure_path, f"energy_T{T}.pdf")
    plot_energy(energies, T, energy_figure_name)
