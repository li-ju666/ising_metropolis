from matplotlib import pyplot as plt

# Speedup
N = [10, 25, 50, 100, 150, 200]
naive_time = [2.133035182952881, 13.89303708076477, 52.33568000793457, 210.31251502037048, 480.1773362159729, 838.8989520072937]
para_time = [0.9170360565185547, 1.5476830005645752, 3.6387531757354736, 12.00826120376587, 27.0107741355896, 47.538570165634155]

speedup = [naive_time[i] / para_time[i] for i in range(len(N))]
plt.plot(N, speedup, "-o", markersize=3)
plt.xlabel("$N$")
plt.ylabel("Speedup")
plt.title("Speedup of Parallelized Implementation")

plt.savefig("./figures/speedup.pdf", dpi=300)
plt.close()
