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

TEST_LABELS = {
    "baseline": "Baseline",
    "test_1": "Test 1: fear only",
    "test_2": "Test 2: cluster only",
    "test_3": "Test 3: wolf fear + protection",
    "test_4": "Test 4: safety seeking",
    "test_5": "Test 5: realistic trigger",
}

FITNESS_LABELS = {
    "current": "Current",
    "survival_bonus": "Survival bonus",
    "strong_survival": "Strong survival",
}

TEST_COLORS = {
    "baseline": "#4d4d4d",
    "test_1": "#1f77b4",
    "test_2": "#2ca02c",
    "test_3": "#9467bd",
    "test_4": "#ff7f0e",
    "test_5": "#d62728",
}

FITNESS_LINESTYLES = {
    "current": "-",
    "survival_bonus": "--",
    "strong_survival": ":",
}

SUMMARY_METRICS = [
    "avg_sheep_fitness",
    "avg_wolf_fitness",
    "survival_rate",
    "grass_per_sheep",
    "protected_time_fraction",
    "protected_alive_fraction_end",
    "mean_alive_cluster_size_end",
    "largest_alive_cluster_size_end",
    "wolf_kills",
    "objective_survival_foraging_score",
    "objective_cluster_survival_score",
]


@dataclass
class ResultSeries:
    path: Path
    scenario: str
    test: str
    fitness: str
    rows: list[dict[str, float | str]]

    @property
    def label(self) -> str:
        test_label = TEST_LABELS.get(self.test, self.test)
        fitness_label = FITNESS_LABELS.get(self.fitness, self.fitness)
        return f"{test_label} / {fitness_label}"

    @property
    def color(self) -> str:
        return TEST_COLORS.get(self.test, "#333333")

    @property
    def linestyle(self) -> str:
        return FITNESS_LINESTYLES.get(self.fitness, "-")


def load_results() -> list[ResultSeries]:
    paths = sorted(RESULTS_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No herd result CSVs found in {RESULTS_DIR.resolve()}")

    parsed = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                converted = {}
                for key, value in row.items():
                    if key in {"scenario", "test", "fitness"}:
                        converted[key] = value
                    else:
                        try:
                            converted[key] = float(value)
                        except ValueError:
                            converted[key] = value
                if "generation" in converted:
                    converted["generation"] = int(converted["generation"]) + 1
                rows.append(converted)

        if not rows:
            continue

        first = rows[0]
        scenario = str(first.get("scenario", path.stem))
        test = str(first.get("test", infer_test_key(scenario)))
        fitness = str(first.get("fitness", "current"))
        parsed.append(ResultSeries(path, scenario, test, fitness, rows))

    if any("__fit_" in series.path.stem for series in parsed):
        parsed = [series for series in parsed if "__fit_" in series.path.stem]

    if not parsed:
        raise FileNotFoundError(f"No readable herd result CSVs found in {RESULTS_DIR.resolve()}")

    return sorted(parsed, key=lambda s: (s.test, s.fitness))


def infer_test_key(scenario: str) -> str:
    if scenario.startswith("baseline"):
        return "baseline"
    for key in TEST_LABELS:
        if scenario.startswith(key):
            return key
    return scenario


def available_metric(series: ResultSeries, metric: str) -> bool:
    return metric in series.rows[0]


def moving_average(values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values

    smoothed = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed[i] = values[start : i + 1].mean()
    return smoothed


def metric_arrays(series: ResultSeries, metric: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([row["generation"] for row in series.rows], dtype=float)
    y = np.array([row[metric] for row in series.rows], dtype=float)
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
    series_list: list[ResultSeries],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    available = [series for series in series_list if available_metric(series, metric)]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(13, 8))

    for series in available:
        x, y = metric_arrays(series, metric)
        ax.plot(x, y, color=series.color, alpha=0.10, linewidth=0.8)
        ax.plot(
            x,
            moving_average(y),
            color=series.color,
            linestyle=series.linestyle,
            linewidth=2.1,
            label=series.label,
        )

    style_axis(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    save(fig, filename)


def plot_final_window_summary(series_list: list[ResultSeries], metric: str, ylabel: str, filename: str) -> None:
    available = [series for series in series_list if available_metric(series, metric)]
    if not available:
        return

    labels = [series.label for series in available]
    means = []
    colors = []
    hatches = []

    for series in available:
        _, y = metric_arrays(series, metric)
        means.append(y[-FINAL_WINDOW:].mean())
        colors.append(series.color)
        hatches.append({"current": "", "survival_bonus": "//", "strong_survival": ".."}.get(series.fitness, ""))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, means, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    style_axis(ax, f"Final {FINAL_WINDOW}-Generation Mean: {ylabel}", ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    save(fig, filename)


def write_summary_csv(series_list: list[ResultSeries]) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / "summary_stats.csv"
    metrics = [
        metric
        for metric in SUMMARY_METRICS
        if any(available_metric(series, metric) for series in series_list)
    ]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["scenario", "test", "fitness", "result_file", "generations"]
        for metric in metrics:
            header.extend([
                f"final_{metric}",
                f"final_{FINAL_WINDOW}_mean_{metric}",
                f"best_{metric}",
            ])
        writer.writerow(header)

        for series in series_list:
            row = [
                series.scenario,
                series.test,
                series.fitness,
                series.path.name,
                len(series.rows),
            ]
            for metric in metrics:
                if not available_metric(series, metric):
                    row.extend(["", "", ""])
                    continue

                _, y = metric_arrays(series, metric)
                row.extend([y[-1], y[-FINAL_WINDOW:].mean(), y.max()])
            writer.writerow(row)


def final_window_mean(series: ResultSeries, metric: str) -> float:
    _, y = metric_arrays(series, metric)
    return float(y[-FINAL_WINDOW:].mean())


def final_window_mean_when_alive(series: ResultSeries, metric: str) -> float:
    final_rows = series.rows[-FINAL_WINDOW:]
    values = [
        float(row[metric])
        for row in final_rows
        if float(row.get("sheep_alive", 0.0)) > 0.0
    ]
    if not values:
        return 0.0
    return float(np.array(values, dtype=float).mean())


def report_table_rows(series_list: list[ResultSeries]) -> list[dict[str, str | float]]:
    available = [
        series
        for series in series_list
        if all(
            available_metric(series, metric)
            for metric in [
                "survival_rate",
                "protected_time_fraction",
                "mean_alive_cluster_size_end",
                "grass_per_sheep",
                "wolf_kills",
                "objective_cluster_survival_score",
            ]
        )
    ]

    ranked = sorted(
        available,
        key=lambda series: final_window_mean(series, "objective_cluster_survival_score"),
        reverse=True,
    )

    rows = []
    for rank, series in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank,
                "test": TEST_LABELS.get(series.test, series.test),
                "fitness": FITNESS_LABELS.get(series.fitness, series.fitness),
                "survival_pct": final_window_mean(series, "survival_rate") * 100.0,
                "protected_time_pct": final_window_mean(series, "protected_time_fraction") * 100.0,
                "mean_end_cluster_size_when_alive": final_window_mean_when_alive(
                    series,
                    "mean_alive_cluster_size_end",
                ),
                "grass_per_sheep": final_window_mean(series, "grass_per_sheep"),
                "wolf_kills": final_window_mean(series, "wolf_kills"),
                "cluster_survival_score": final_window_mean(series, "objective_cluster_survival_score"),
            }
        )
    return rows


def write_report_table(series_list: list[ResultSeries]) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    rows = report_table_rows(series_list)

    csv_path = PLOTS_DIR / "report_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "test",
                "fitness",
                "survival_percent",
                "protected_time_percent",
                "mean_end_cluster_size_when_alive",
                "grass_per_sheep",
                "wolf_kills",
                "cluster_survival_score",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["rank"],
                    row["test"],
                    row["fitness"],
                    f"{row['survival_pct']:.1f}",
                    f"{row['protected_time_pct']:.1f}",
                    f"{row['mean_end_cluster_size_when_alive']:.2f}",
                    f"{row['grass_per_sheep']:.2f}",
                    f"{row['wolf_kills']:.2f}",
                    f"{row['cluster_survival_score']:.3f}",
                ]
            )

    markdown_path = PLOTS_DIR / "report_table.md"
    with markdown_path.open("w", newline="") as f:
        f.write("# Herd Experiment Report Table\n\n")
        f.write(
            f"Values are means over the final {FINAL_WINDOW} generations. "
            "Rows are ranked by the cluster-survival score, defined as "
            "survival rate x protected time fraction. Mean end cluster size "
            "is averaged only across generations where at least one sheep survived.\n\n"
        )
        f.write(
            "| Rank | Test condition | Fitness treatment | Survival (%) | "
            "Protected time (%) | Mean end cluster size when alive | Grass/sheep | "
            "Wolf kills | Cluster-survival score |\n"
        )
        f.write("|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['rank']} | {row['test']} | {row['fitness']} | "
                f"{row['survival_pct']:.1f} | {row['protected_time_pct']:.1f} | "
                f"{row['mean_end_cluster_size_when_alive']:.2f} | {row['grass_per_sheep']:.2f} | "
                f"{row['wolf_kills']:.2f} | {row['cluster_survival_score']:.3f} |\n"
            )

    latex_path = PLOTS_DIR / "report_table.tex"
    with latex_path.open("w", newline="") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Final 50-generation herd experiment performance summary. Rows are ranked by survival rate multiplied by protected time fraction. Mean end cluster size is averaged only over generations with at least one surviving sheep.}\n")
        f.write("\\begin{tabular}{rllrrrrrr}\n")
        f.write("\\hline\n")
        f.write("Rank & Test & Fitness & Survival (\\%) & Protected (\\%) & Cluster size when alive & Grass/sheep & Wolf kills & Score \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(
                f"{row['rank']} & {row['test']} & {row['fitness']} & "
                f"{row['survival_pct']:.1f} & {row['protected_time_pct']:.1f} & "
                f"{row['mean_end_cluster_size_when_alive']:.2f} & {row['grass_per_sheep']:.2f} & "
                f"{row['wolf_kills']:.2f} & {row['cluster_survival_score']:.3f} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main() -> None:
    series_list = load_results()

    plot_metric_comparison(
        series_list,
        "avg_sheep_fitness",
        "Average sheep fitness",
        "Sheep Fitness Across Scenario/Fitness Tests",
        "sheep_fitness_comparison.png",
    )
    plot_metric_comparison(
        series_list,
        "avg_wolf_fitness",
        "Average wolf fitness",
        "Wolf Fitness Across Scenario/Fitness Tests",
        "wolf_fitness_comparison.png",
    )
    plot_metric_comparison(
        series_list,
        "survival_rate",
        "Survival rate",
        "Objective Survival Rate Across Tests",
        "survival_rate_comparison.png",
    )
    plot_metric_comparison(
        series_list,
        "protected_time_fraction",
        "Protected time fraction",
        "Objective Clustering Behaviour Across Tests",
        "protected_time_fraction_comparison.png",
    )
    plot_metric_comparison(
        series_list,
        "grass_per_sheep",
        "Grass eaten per sheep",
        "Foraging Performance Across Tests",
        "grass_per_sheep_comparison.png",
    )
    plot_metric_comparison(
        series_list,
        "objective_cluster_survival_score",
        "Survival rate x protected time fraction",
        "Composite Clustering-Survival Objective",
        "objective_cluster_survival_score.png",
    )

    plot_final_window_summary(
        series_list,
        "survival_rate",
        "Survival rate",
        "final_window_survival_rate.png",
    )
    plot_final_window_summary(
        series_list,
        "protected_time_fraction",
        "Protected time fraction",
        "final_window_protected_time_fraction.png",
    )
    plot_final_window_summary(
        series_list,
        "objective_cluster_survival_score",
        "Survival x protected time",
        "final_window_cluster_survival_objective.png",
    )
    write_summary_csv(series_list)
    write_report_table(series_list)

    print(f"Loaded {len(series_list)} result files")
    print(f"Wrote plots to {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
