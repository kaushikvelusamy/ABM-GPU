import csv
import json
from pathlib import Path


def human_readable_size_from_bytes(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def infer_total_output_size_bytes(run_config):
    if not run_config:
        return None

    candidate_keys = [
        "summary_csv",
        "agent_history_csv",
        "location_history_csv",
        "run_config_json",
        "plot_out",
    ]
    total = 0
    found_any = False
    for key in candidate_keys:
        path = run_config.get(key)
        if not path:
            continue
        candidate = Path(path)
        if candidate.exists():
            total += candidate.stat().st_size
            found_any = True

    if not found_any:
        return None
    return total


def read_summary_csv(path):
    steps = []
    susceptible = []
    infected = []
    recovered = []
    occupied_locations = []
    max_occupancy = []
    moved_agents = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "step",
            "susceptible",
            "infected",
            "recovered",
            "occupied_locations",
            "max_occupancy",
            "moved_agents",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        for row in reader:
            steps.append(int(row["step"]))
            susceptible.append(float(row["susceptible"]))
            infected.append(float(row["infected"]))
            recovered.append(float(row["recovered"]))
            occupied_locations.append(float(row["occupied_locations"]))
            max_occupancy.append(float(row["max_occupancy"]))
            moved_agents.append(float(row["moved_agents"]))

    return {
        "steps": steps,
        "susceptible": susceptible,
        "infected": infected,
        "recovered": recovered,
        "occupied_locations": occupied_locations,
        "max_occupancy": max_occupancy,
        "moved_agents": moved_agents,
    }


def default_plot_path(summary_csv_path):
    summary_csv_path = Path(summary_csv_path)
    name = summary_csv_path.name
    if name.endswith("_summary.csv"):
        return summary_csv_path.with_name(name[:-12] + "_summary.png")
    return summary_csv_path.with_suffix(".png")


def infer_run_config_path(summary_csv_path):
    summary_csv_path = Path(summary_csv_path)
    name = summary_csv_path.name
    if name.endswith("_summary.csv"):
        candidate = summary_csv_path.with_name(name[:-12] + "_run_config.json")
        if candidate.exists():
            return candidate
    return None


def load_run_config(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_annotation_lines(series, run_config):
    peak_i = max(series["infected"]) if series["infected"] else 0.0
    peak_step = series["steps"][series["infected"].index(peak_i)] if series["infected"] else 0
    final_s = series["susceptible"][-1] if series["susceptible"] else 0.0
    final_i = series["infected"][-1] if series["infected"] else 0.0
    final_r = series["recovered"][-1] if series["recovered"] else 0.0

    parameter_lines = []

    if run_config:
        runtime_seconds = run_config.get("runtime_seconds")
        peak_memory_mb = run_config.get("peak_memory_mb")
        total_output_size_bytes = run_config.get("total_output_size_bytes")
        if total_output_size_bytes is None:
            total_output_size_bytes = infer_total_output_size_bytes(run_config)
        parameter_lines.extend(
            [
                "Model Parameters",
                f"Execution: {run_config.get('execution', run_config.get('mode', 'unknown'))}",
                f"Grid: {run_config.get('rows', '?')} x {run_config.get('cols', '?')}",
                f"Locations: {run_config.get('num_locations', '?')}",
                f"Number of agents: {run_config.get('num_agents', '?')}",
                f"Initial infected: {run_config.get('infected0', '?')}",
                f"beta={run_config.get('beta', '?')}, gamma={run_config.get('gamma', '?')}",
                f"Simulation steps: {run_config.get('steps', '?')}",
                f"Move probability: {run_config.get('move_prob', '?')}",
                f"Infection neighborhood: {run_config.get('infection_neighborhood', '?')}",
                f"Movement neighborhood: {run_config.get('movement_neighborhood', '?')}",
                f"Seed: {run_config.get('seed', '?')}",
                (
                    f"CPU runtime: {float(runtime_seconds):.3f} s"
                    if runtime_seconds is not None
                    else "CPU runtime: unavailable"
                ),
                (
                    f"CPU peak memory: {float(peak_memory_mb):.3f} MB"
                    if peak_memory_mb is not None
                    else "CPU peak memory: unavailable"
                ),
                (
                    f"Total storage: {human_readable_size_from_bytes(float(total_output_size_bytes))}"
                    if total_output_size_bytes is not None
                    else "Total storage: unavailable"
                ),
            ]
        )

    result_lines = [
        "Model Results",
        f"Final susceptible: {final_s:.0f}",
        f"Final infected: {final_i:.0f}",
        f"Final recovered: {final_r:.0f}",
        f"Peak infected: {peak_i:.0f} at day {peak_step}",
    ]

    return parameter_lines, result_lines


def plot_summary(
    csv_path,
    out_png=None,
    title="Moving-Agent SIR Summary",
    x_label="Step",
    config_json=None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib in this Python environment."
        ) from exc

    series = read_summary_csv(csv_path)
    run_config = load_run_config(config_json or infer_run_config_path(csv_path))
    if out_png is None:
        out_png = default_plot_path(csv_path)

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        "susceptible": "#1f3b75",
        "infected": "#9b2d26",
        "recovered": "#2f5d3a",
    }

    ax.plot(
        series["steps"],
        series["susceptible"],
        label="S",
        color=colors["susceptible"],
        linewidth=2.6,
        alpha=0.95,
    )
    ax.plot(
        series["steps"],
        series["infected"],
        label="I",
        color=colors["infected"],
        linewidth=2.6,
        alpha=0.95,
    )
    ax.plot(
        series["steps"],
        series["recovered"],
        label="R",
        color=colors["recovered"],
        linewidth=2.6,
        alpha=0.95,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of People")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="center left", frameon=True)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(right=0.72)

    parameter_lines, result_lines = build_annotation_lines(series, run_config)
    if parameter_lines:
        fig.text(
            0.75,
            0.88,
            "\n".join(parameter_lines),
            transform=fig.transFigure,
            fontsize=9,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.9},
        )
    fig.text(
        0.75,
        0.42,
        "\n".join(result_lines),
        transform=fig.transFigure,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.9},
    )

    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)
