# visualize the lattice
import numpy as np
from matplotlib import pyplot as plt


def read(T, idx, interval=50, take=None):
    data = np.load(f"./data/crit_temp/{idx}.npz")
    lattices = data["lattices"]
    lattices = lattices if not take else lattices[:take]
    lattices = lattices[::interval]
    return lattices


if __name__ == "__main__":
    Ts1 = np.linspace(1, 2, 10)
    Ts2 = np.linspace(2, 2.5, 30)
    Ts3 = np.linspace(2.5, 3.5, 10)

    Ts = np.concatenate([Ts1, Ts2, Ts3])

    lattices = []
    temperatures = []

    for idx, t in enumerate(Ts[10:]):
        data = read(t, idx)
        ts = np.ones(len(data)) * t
        lattices.append(data)
        temperatures.append(ts)

    lattices = np.concatenate(lattices)
    temperatures = np.concatenate(temperatures)

    # animate the lattices
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])

    im = ax.imshow(lattices[0], vmin=-1, vmax=1)

    def animate(i):
        im.set_data(lattices[i])
        ax.set_title(f"T = {temperatures[i]:.2f}")

    from matplotlib.animation import FuncAnimation

    anim = FuncAnimation(fig, animate, frames=len(lattices), interval=100)
    anim.save("phase_transition.gif", dpi=100, writer="imagemagick")
    plt.show()
