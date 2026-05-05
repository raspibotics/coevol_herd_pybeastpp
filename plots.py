import pandas as pd
import matplotlib.pyplot as plt

# Load one scenario (or loop later for multiple)
df = pd.read_csv("results/test_5_cluster_fear_wolf_and_unsafe.csv")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Left Y-axis (Sheep)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Sheep Fitness")
ax1.plot(
    df["generation"],
    df["avg_sheep_fitness"],
    color="blue",
    label="Sheep Fitness"
)
ax1.tick_params(axis='y')

# Right Y-axis (Wolf)
ax2 = ax1.twinx()
ax2.set_ylabel("Wolf Fitness")
ax2.plot(
    df["generation"],
    df["avg_wolf_fitness"],
    color="red",
    label="Wolf Fitness"
)
ax2.tick_params(axis='y')

# Title
plt.title("Sheep vs Wolf Fitness Across Generations")

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.grid(True)
plt.tight_layout()
plt.savefig("dual_axis_fitness.png", dpi=300)
plt.show()