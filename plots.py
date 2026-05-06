import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "coevol_herd_plot_cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
SMOOTHING_WINDOW = 15
FINAL_WINDOW = 50


@dataclass(frozen=True)
class ScenarioPlotConfig:
    slug: str
    label: str
    color: str


SCENARIOS = [
    ScenarioPlotConfig(
        "baseline_no_cluster_no_fear",
        "Baseline",
        "#4d4d4d",
    ),
    ScenarioPlotConfig(
        "test_1_no_cluster_fear_wolf_seen",
        "Test 1: fear only",
        "#1f77b4",
    ),
    ScenarioPlotConfig(
        "test_2_cluster_no_fear",
        "Test 2: cluster only",
        "#2ca02c",
    ),
    ScenarioPlotConfig(
        "test_3_cluster_fear_wolf_seen",
        "Test 3: wolf fear + protection",
        "#9467bd",
    ),
    ScenarioPlotConfig(
        "test_4_cluster_fear_unsafe",
        "Test 4: safety seeking",
        "#ff7f0e",
    ),
    ScenarioPlotConfig(
        "test_5_cluster_fear_wolf_and_unsafe",
        "Test 5: realistic trigger",
        "#d62728",
    ),
]


def load_results() -> dict[ScenarioPlotConfig, list[dict[str, float]]]:
    loaded = {}

    for scenario in SCENARIOS:
        path = RESULTS_DIR / f"{scenario.slug}.csv"
        if not path.exists():
            continue

        with path.open(newline="") as f:
            rows = []
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "generation": int(row["generation"]) + 1,
                        "avg_sheep_fitness": float(row["avg_sheep_fitness"]),
                        "avg_wolf_fitness": float(row["avg_wolf_fitness"]),
                    }
                )

        if rows:
            loaded[scenario] = rows

    if not loaded:
        raise FileNotFoundError(f"No herd result CSVs found in {RESULTS_DIR.resolve()}")

    return loaded


def moving_average(values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values

    smoothed = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed[i] = values[start : i + 1].mean()
    return smoothed


def metric_arrays(rows: list[dict[str, float]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([row["generation"] for row in rows], dtype=float)
    y = np.array([row[metric] for row in rows], dtype=float)
    return x, y


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close(fig)


def plot_metric_comparison(
    data: dict[ScenarioPlotConfig, list[dict[str, float]]],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for scenario, rows in data.items():
        x, y = metric_arrays(rows, metric)
        ax.plot(x, y, color=scenario.color, alpha=0.16, linewidth=0.8)
        ax.plot(
            x,
            moving_average(y),
            color=scenario.color,
            linewidth=2.2,
            label=scenario.label,
        )

    style_axis(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=9)
    save(fig, filename)


def plot_dual_axis_by_test(data: dict[ScenarioPlotConfig, list[dict[str, float]]]) -> None:
    scenarios = list(data)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 3.2 * len(scenarios)), sharex=True)

    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        rows = data[scenario]
        x, sheep = metric_arrays(rows, "avg_sheep_fitness")
        _, wolf = metric_arrays(rows, "avg_wolf_fitness")

        sheep_line = ax.plot(
            x,
            moving_average(sheep),
            color="#1f77b4",
            linewidth=2.0,
            label="Sheep fitness",
        )
        ax.set_ylabel("Sheep fitness")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)

        ax2 = ax.twinx()
        wolf_line = ax2.plot(
            x,
            moving_average(wolf),
            color="#d62728",
            linewidth=2.0,
            label="Wolf fitness",
        )
        ax2.set_ylabel("Wolf fitness")
        ax2.spines["top"].set_visible(False)

        ax.set_title(scenario.label, fontsize=12, loc="left")
        lines = sheep_line + wolf_line
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, frameon=False, fontsize=9, loc="upper left")

    axes[-1].set_xlabel("Generation")
    fig.suptitle("Sheep and Wolf Fitness by Test", fontsize=15, y=1.0)
    save(fig, "sheep_vs_wolf_by_test.png")


def plot_final_window_summary(data: dict[ScenarioPlotConfig, list[dict[str, float]]]) -> None:
    labels = [scenario.label for scenario in data]
    sheep_means = []
    wolf_means = []

    for rows in data.values():
        sheep = np.array([row["avg_sheep_fitness"] for row in rows], dtype=float)
        wolf = np.array([row["avg_wolf_fitness"] for row in rows], dtype=float)
        sheep_means.append(sheep[-FINAL_WINDOW:].mean())
        wolf_means.append(wolf[-FINAL_WINDOW:].mean())

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.bar(x, sheep_means, color=[scenario.color for scenario in data])
    style_axis(
        ax1,
        f"Mean Sheep Fitness Over Final {FINAL_WINDOW} Generations",
        "Sheep fitness",
    )

    ax2.bar(x, wolf_means, color=[scenario.color for scenario in data])
    style_axis(
        ax2,
        f"Mean Wolf Fitness Over Final {FINAL_WINDOW} Generations",
        "Wolf fitness",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")

    save(fig, "final_window_mean_fitness.png")


def plot_sheep_wolf_ratio(data: dict[ScenarioPlotConfig, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for scenario, rows in data.items():
        x, sheep = metric_arrays(rows, "avg_sheep_fitness")
        _, wolf = metric_arrays(rows, "avg_wolf_fitness")
        ratio = sheep / np.maximum(wolf, 1e-9)
        ax.plot(
            x,
            moving_average(ratio),
            color=scenario.color,
            linewidth=2.0,
            label=scenario.label,
        )

    style_axis(
        ax,
        "Sheep-to-Wolf Fitness Ratio",
        "Sheep fitness / wolf fitness",
    )
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "sheep_wolf_fitness_ratio.png")


def write_summary_csv(data: dict[ScenarioPlotConfig, list[dict[str, float]]]) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / "summary_stats.csv"

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "generations",
                "final_sheep_fitness",
                "final_wolf_fitness",
                f"final_{FINAL_WINDOW}_mean_sheep_fitness",
                f"final_{FINAL_WINDOW}_mean_wolf_fitness",
                "best_sheep_fitness",
                "best_wolf_fitness",
            ]
        )

        for scenario, rows in data.items():
            sheep = np.array([row["avg_sheep_fitness"] for row in rows], dtype=float)
            wolf = np.array([row["avg_wolf_fitness"] for row in rows], dtype=float)
            writer.writerow(
                [
                    scenario.label,
                    len(rows),
                    sheep[-1],
                    wolf[-1],
                    sheep[-FINAL_WINDOW:].mean(),
                    wolf[-FINAL_WINDOW:].mean(),
                    sheep.max(),
                    wolf.max(),
                ]
            )


def main() -> None:
    data = load_results()

    plot_metric_comparison(
        data,
        "avg_sheep_fitness",
        "Average sheep fitness",
        "Sheep Fitness Across Herd Tests",
        "sheep_fitness_comparison.png",
    )
    plot_metric_comparison(
        data,
        "avg_wolf_fitness",
        "Average wolf fitness",
        "Wolf Fitness Across Herd Tests",
        "wolf_fitness_comparison.png",
    )
    plot_dual_axis_by_test(data)
    plot_final_window_summary(data)
    plot_sheep_wolf_ratio(data)
    write_summary_csv(data)

    print(f"Wrote plots to {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
