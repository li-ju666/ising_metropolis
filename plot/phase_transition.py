import numpy as np
from matplotlib import pyplot as plt


def read(idx):
    data = np.load(f"./data/crit_temp/{idx}.npz")
    magnetizations = data["magnetizations"]
    energies = data["energies"]
    return magnetizations, energies


def estimate(data, delta_t, tau):
    eq_idx = int(5e3)
    data = data[eq_idx:]
    t = data.shape[0]
    hat_m = np.mean(data)
    hat_m2 = np.mean(data**2)
    error = np.sqrt(
        (hat_m2 - hat_m**2)*(1+2*tau/delta_t)/(t-1))
    return hat_m, error


if __name__ == "__main__":
    indices = range(50)

    mag_means = []
    mag_stds = []
    for idx in indices:
        mag, _ = read(idx)

        delta_t = 1e4
        tau = 5e6

        hat_m, error = estimate(mag, delta_t, tau)
        mag_means.append(hat_m)
        mag_stds.append(error)

    mag_means = np.array(mag_means)
    mag_stds = np.array(mag_stds)

    # plot the magnetization w.r.t temperature
    # temperatures = np.linspace(1.0, 4.0, 50)
    # temperatures = temperatures[indices]
    Ts1 = np.linspace(1.0, 2.0, 10)
    Ts2 = np.linspace(2.0, 2.5, 30)
    Ts3 = np.linspace(2.5, 4.0, 10)

    temperatures = np.concatenate([Ts1, Ts2, Ts3])

    plt.figure()

    plt.plot(temperatures, mag_means, 'o-', markersize=2)
    plt.fill_between(temperatures, mag_means - mag_stds,
                     mag_means + mag_stds, alpha=0.3)
    plt.xlabel("Temperature $T(K)$")
    plt.ylabel("Magnetization")
    plt.title("Magnetization vs Temperature")
    plt.savefig("./figures/mag_vs_T.pdf", dpi=300)
    plt.close()
