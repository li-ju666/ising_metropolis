import numpy as np
from matplotlib import pyplot as plt
import os


def read(T, random_seed):
    data = np.load(f"./data/T{T}/seed{random_seed}.npz")
    magnetizations = data["magnetizations"]
    energies = data["energies"]
    return magnetizations, energies


# compute the time-displaced autocorrelation
def autocorrelation(data, max_lag):
    N = len(data)
    mean = np.mean(data)
    var = np.var(data)
    cor = np.zeros(max_lag)
    for lag in range(max_lag):
        cor[lag] = np.sum((data[:N-lag]-mean)*(data[lag:]-mean))/(N-lag)/var
    return cor


def plot_autocor(mag_cor, energy_cor, T, name=None):
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mag_cor, alpha=0.8, label="Magnetization")
    ax.plot(energy_cor, alpha=0.8, label="Energy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Autocorrealtion")
    ax.set_title(f"Autocorrelation at $T = {T}$")
    # set limits
    ax.set_ylim(-0.2, 1.1)
    ax.legend()
    if name:
        plt.savefig(name, dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    T = 2.4
    seed = 1

    mags = []
    energies = []
    mag, energy = read(T, seed)
    eq_idx = int(5e3)
    mag, energy = mag[eq_idx:], energy[eq_idx:]

    # compute the autocorrelation for different lags
    max_lag = 2000
    mag_cors = autocorrelation(mag, max_lag)
    energy_cors = autocorrelation(energy, max_lag)

    # plot the magnetization
    figure_path = "./figures/"
    os.makedirs(figure_path, exist_ok=True)

    figure_name = os.path.join(figure_path, f"autocor_T{T}.pdf")
    plot_autocor(mag_cors, energy_cors, T, figure_name)
