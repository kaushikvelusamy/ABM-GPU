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

    run_dir = run_config.get("run_dir")
    if run_dir:
        run_dir_path = Path(run_dir)
        if run_dir_path.exists():
            for child_dir, pattern in (
                (run_dir_path / "agent_history", "rank*_agent_history.csv"),
                (run_dir_path / "location_history", "rank*_location_history.csv"),
            ):
                if child_dir.exists():
                    for candidate in child_dir.glob(pattern):
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
    return summary_csv_path.with_suffix(".png")


def infer_run_config_path(summary_csv_path):
    summary_csv_path = Path(summary_csv_path)
    candidate = summary_csv_path.with_name("run_config.json")
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
        runtime_seconds = run_config.get("total_wall_time_seconds", run_config.get("distributed_runtime_seconds"))
        host_memory_per_rank_mb = run_config.get("host_memory_per_rank_mb", run_config.get("peak_rank_memory_mb"))
        total_host_memory_mb = run_config.get("total_host_memory_mb")
        total_gpu_compute_seconds = run_config.get("total_gpu_compute_seconds")
        gpu_compute_time_max_rank_seconds = run_config.get(
            "gpu_compute_time_max_rank_seconds",
            total_gpu_compute_seconds,
        )
        gpu_compute_fraction_of_wall = run_config.get("gpu_compute_fraction_of_wall")
        gpu_memory_per_rank_mb = run_config.get("gpu_memory_per_rank_mb", run_config.get("peak_gpu_memory_mb"))
        total_gpu_memory_mb = run_config.get("total_gpu_memory_mb")
        total_communication_seconds = run_config.get("total_communication_seconds")
        communication_time_max_rank_seconds = run_config.get(
            "communication_time_max_rank_seconds",
            total_communication_seconds,
        )
        communication_fraction_of_wall = run_config.get("communication_fraction_of_wall")
        total_output_io_seconds = run_config.get("total_output_io_seconds")
        plot_generation_seconds = run_config.get("plot_generation_seconds")
        total_output_size_bytes = run_config.get("total_output_size_bytes")
        if total_output_size_bytes is None:
            total_output_size_bytes = infer_total_output_size_bytes(run_config)
        parameter_lines.extend(
            [
                "Model Parameters",
                f"Execution: {run_config.get('execution', run_config.get('mode', 'unknown'))}",
                f"Backend: {run_config.get('backend', '?')}",
                f"Total nodes: {run_config.get('total_nodes', '?')}",
                f"Total MPI ranks: {run_config.get('total_mpi_ranks', run_config.get('world_size', '?'))}",
                f"Total GPUs used: {run_config.get('total_gpus_used', '?')}",
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
                    f"Total wall time: {float(runtime_seconds):.3f} s"
                    if runtime_seconds is not None
                    else "Total wall time: unavailable"
                ),
                (
                    f"Host memory per rank: {float(host_memory_per_rank_mb):.3f} MB"
                    if host_memory_per_rank_mb is not None
                    else "Host memory per rank: unavailable"
                ),
                (
                    f"Total host memory: {float(total_host_memory_mb):.3f} MB"
                    if total_host_memory_mb is not None
                    else "Total host memory: unavailable"
                ),
                (
                    (
                        f"GPU compute time on slowest rank: {float(gpu_compute_time_max_rank_seconds):.3f} s "
                        f"({float(gpu_compute_fraction_of_wall) * 100.0:.1f}% of wall time)"
                    )
                    if gpu_compute_time_max_rank_seconds is not None and gpu_compute_fraction_of_wall is not None
                    else (
                        f"GPU compute time on slowest rank: {float(gpu_compute_time_max_rank_seconds):.3f} s"
                        if gpu_compute_time_max_rank_seconds is not None
                        else "GPU compute time on slowest rank: unavailable"
                    )
                ),
                (
                    f"GPU compute time (all ranks summed): {float(total_gpu_compute_seconds):.3f} s"
                    if total_gpu_compute_seconds is not None
                    else "GPU compute time (all ranks summed): unavailable"
                ),
                (
                    f"GPU memory per rank: {float(gpu_memory_per_rank_mb):.3f} MB"
                    if gpu_memory_per_rank_mb is not None
                    else "GPU memory per rank: unavailable"
                ),
                (
                    f"Total GPU memory: {float(total_gpu_memory_mb):.3f} MB"
                    if total_gpu_memory_mb is not None
                    else "Total GPU memory: unavailable"
                ),
                (
                    (
                        f"Communication time on slowest rank: {float(communication_time_max_rank_seconds):.3f} s "
                        f"({float(communication_fraction_of_wall) * 100.0:.1f}% of wall time)"
                    )
                    if communication_time_max_rank_seconds is not None and communication_fraction_of_wall is not None
                    else (
                        f"Communication time on slowest rank: {float(communication_time_max_rank_seconds):.3f} s"
                        if communication_time_max_rank_seconds is not None
                        else "Communication time on slowest rank: unavailable"
                    )
                ),
                (
                    f"Communication time (all ranks summed): {float(total_communication_seconds):.3f} s"
                    if total_communication_seconds is not None
                    else "Communication time (all ranks summed): unavailable"
                ),
                (
                    f"Total output I/O time: {float(total_output_io_seconds):.3f} s"
                    if total_output_io_seconds is not None
                    else "Total output I/O time: unavailable"
                ),
                (
                    f"Plot generation time: {float(plot_generation_seconds):.3f} s"
                    if plot_generation_seconds is not None
                    else "Plot generation time: unavailable"
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


def plot_summary(csv_path, out_png=None, title="Distributed Moving-Agent SIR", x_label="Day", config_json=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib in this Python environment.") from exc

    series = read_summary_csv(csv_path)
    run_config = load_run_config(config_json or infer_run_config_path(csv_path))
    if out_png is None:
        out_png = default_plot_path(csv_path)

    fig, (ax, side_ax) = plt.subplots(
        1,
        2,
        figsize=(16, 9),
        gridspec_kw={"width_ratios": [3.8, 1.9]},
    )
    colors = {
        "susceptible": "#1f3b75",
        "infected": "#9b2d26",
        "recovered": "#2f5d3a",
    }

    ax.plot(series["steps"], series["susceptible"], label="S", color=colors["susceptible"], linewidth=2.6, alpha=0.95)
    ax.plot(series["steps"], series["infected"], label="I", color=colors["infected"], linewidth=2.6, alpha=0.95)
    ax.plot(series["steps"], series["recovered"], label="R", color=colors["recovered"], linewidth=2.6, alpha=0.95)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of People")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="center left", frameon=True)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    side_ax.axis("off")
    fig.subplots_adjust(wspace=0.08)

    parameter_lines, result_lines = build_annotation_lines(series, run_config)
    if parameter_lines:
        side_ax.text(
            0.02,
            0.98,
            "\n".join(parameter_lines),
            transform=side_ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.9},
        )
    parameter_box_height = 0.038 * max(len(parameter_lines), 1) + 0.06
    result_y = max(0.06, 0.98 - parameter_box_height - 0.05)
    side_ax.text(
        0.02,
        result_y,
        "\n".join(result_lines),
        transform=side_ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.9},
    )

    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)
