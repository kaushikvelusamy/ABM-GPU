import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:
    resource = None

from moving_sir_plotting import plot_summary


SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

STATE_NAME = {
    SUSCEPTIBLE: "S",
    INFECTED: "I",
    RECOVERED: "R",
}


def neighborhood_choices(name):
    if name == "same_cell":
        return 0, False
    if name == "same_plus_news":
        return 1, False
    raise ValueError(f"Unsupported infection neighborhood: {name}")


def movement_deltas(name):
    if name == "news":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if name == "stay_news":
        return [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    raise ValueError(f"Unsupported movement neighborhood: {name}")


def build_location_stats(rows, cols, agent_rows, agent_cols, states):
    cell_ids = agent_rows * cols + agent_cols
    n_cells = rows * cols

    occupancy = np.bincount(cell_ids, minlength=n_cells).astype(np.int64)
    s_counts = np.bincount(cell_ids, weights=(states == SUSCEPTIBLE), minlength=n_cells).astype(np.int64)
    i_counts = np.bincount(cell_ids, weights=(states == INFECTED), minlength=n_cells).astype(np.int64)
    r_counts = np.bincount(cell_ids, weights=(states == RECOVERED), minlength=n_cells).astype(np.int64)

    return (
        occupancy.reshape(rows, cols),
        s_counts.reshape(rows, cols),
        i_counts.reshape(rows, cols),
        r_counts.reshape(rows, cols),
    )


def cardinal_neighbor_sum(grid):
    out = np.zeros_like(grid, dtype=np.int64)
    out[1:, :] += grid[:-1, :]
    out[:-1, :] += grid[1:, :]
    out[:, 1:] += grid[:, :-1]
    out[:, :-1] += grid[:, 1:]
    return out


def move_agents(agent_rows, agent_cols, rows, cols, move_prob, movement_rule, rng):
    old_rows = agent_rows.copy()
    old_cols = agent_cols.copy()
    moved = np.zeros(agent_rows.shape[0], dtype=bool)

    deltas = movement_deltas(movement_rule)
    want_move = rng.random(agent_rows.shape[0]) < move_prob

    for idx in np.flatnonzero(want_move):
        valid = []
        for dr, dc in deltas:
            nr = agent_rows[idx] + dr
            nc = agent_cols[idx] + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                valid.append((dr, dc))
        if not valid:
            continue
        dr, dc = valid[rng.integers(len(valid))]
        nr = agent_rows[idx] + dr
        nc = agent_cols[idx] + dc
        moved[idx] = (nr != agent_rows[idx]) or (nc != agent_cols[idx])
        agent_rows[idx] = nr
        agent_cols[idx] = nc

    return old_rows, old_cols, moved


def write_run_metadata(path, config):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def peak_rss_mb():
    if resource is None:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def human_readable_size(path):
    size = Path(path).stat().st_size
    return human_readable_size_from_bytes(size)


def human_readable_size_from_bytes(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def total_output_size_bytes(paths):
    total = 0
    for path in paths:
        if path is None:
            continue
        candidate = Path(path)
        if candidate.exists():
            total += candidate.stat().st_size
    return total


def write_location_rows(writer, step, occupancy, s_counts, i_counts, r_counts):
    occupied = np.argwhere(occupancy > 0)
    for row, col in occupied:
        writer.writerow(
            [
                step,
                int(row),
                int(col),
                int(occupancy[row, col]),
                int(s_counts[row, col]),
                int(i_counts[row, col]),
                int(r_counts[row, col]),
            ]
        )


def write_agent_rows(
    writer,
    step,
    agent_rows,
    agent_cols,
    states,
    old_rows,
    old_cols,
    moved,
    exposure,
    infection_prob,
    became_infected,
    became_recovered,
):
    for agent_id in range(agent_rows.shape[0]):
        writer.writerow(
            [
                step,
                agent_id,
                int(agent_rows[agent_id]),
                int(agent_cols[agent_id]),
                STATE_NAME[int(states[agent_id])],
                int(old_rows[agent_id]),
                int(old_cols[agent_id]),
                int(agent_rows[agent_id]),
                int(agent_cols[agent_id]),
                int(moved[agent_id]),
                int(exposure[agent_id]),
                float(infection_prob[agent_id]),
                int(became_infected[agent_id]),
                int(became_recovered[agent_id]),
            ]
        )


def resolve_output_paths(out_prefix, run_dir=None):
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return (
            run_dir / "summary.csv",
            run_dir / "agent_history.csv",
            run_dir / "location_history.csv",
            run_dir / "run_config.json",
        )

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    return (
        out_prefix.with_name(out_prefix.name + "_summary.csv"),
        out_prefix.with_name(out_prefix.name + "_agent_history.csv"),
        out_prefix.with_name(out_prefix.name + "_location_history.csv"),
        out_prefix.with_name(out_prefix.name + "_run_config.json"),
    )


def run_simulation(
    rows=100,
    cols=100,
    num_agents=10000,
    infected0=10,
    beta=0.3,
    gamma=0.05,
    steps=30,
    move_prob=0.5,
    infection_neighborhood="same_plus_news",
    movement_neighborhood="stay_news",
    seed=0,
    out_prefix="sir_moving_agents",
    run_dir=None,
    plot_summary_png=False,
    plot_out=None,
):
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")
    if num_agents < 0:
        raise ValueError("num_agents must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0.0 <= move_prob <= 1.0:
        raise ValueError("move_prob must be in [0, 1]")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    num_agents = int(num_agents)
    infected0 = min(int(infected0), num_agents)

    summary_csv, agent_csv, location_csv, metadata_json = resolve_output_paths(out_prefix, run_dir=run_dir)

    agent_rows = rng.integers(0, rows, size=num_agents, dtype=np.int64)
    agent_cols = rng.integers(0, cols, size=num_agents, dtype=np.int64)
    states = np.full(num_agents, SUSCEPTIBLE, dtype=np.int8)

    if infected0 > 0:
        infected_ids = rng.choice(num_agents, size=infected0, replace=False)
        states[infected_ids] = INFECTED

    run_config = {
        "mode": "single_process_cpu",
        "execution": "cpu_only",
        "rows": rows,
        "cols": cols,
        "num_locations": rows * cols,
        "num_agents": num_agents,
        "infected0": infected0,
        "beta": beta,
        "gamma": gamma,
        "steps": steps,
        "move_prob": move_prob,
        "infection_neighborhood": infection_neighborhood,
        "movement_neighborhood": movement_neighborhood,
        "seed": seed,
        "run_dir": str(run_dir) if run_dir else None,
        "plot_summary_png": plot_summary_png,
        "plot_out": str(plot_out) if plot_out else None,
        "run_config_json": str(metadata_json),
        "summary_csv": str(summary_csv),
        "agent_history_csv": str(agent_csv),
        "location_history_csv": str(location_csv),
        "runtime_seconds": None,
        "peak_memory_mb": None,
        "total_output_size_bytes": None,
    }
    write_run_metadata(metadata_json, run_config)

    with open(summary_csv, "w", newline="", encoding="utf-8") as summary_f, open(
        agent_csv, "w", newline="", encoding="utf-8"
    ) as agent_f, open(location_csv, "w", newline="", encoding="utf-8") as location_f:
        summary_writer = csv.writer(summary_f)
        agent_writer = csv.writer(agent_f)
        location_writer = csv.writer(location_f)

        summary_writer.writerow(
            [
                "step",
                "susceptible",
                "infected",
                "recovered",
                "occupied_locations",
                "max_occupancy",
                "moved_agents",
            ]
        )
        agent_writer.writerow(
            [
                "step",
                "agent_id",
                "row",
                "col",
                "state",
                "from_row",
                "from_col",
                "to_row",
                "to_col",
                "moved",
                "exposure",
                "infection_probability",
                "became_infected",
                "became_recovered",
            ]
        )
        location_writer.writerow(
            ["step", "row", "col", "occupancy", "susceptible", "infected", "recovered"]
        )

        occupancy, s_counts, i_counts, r_counts = build_location_stats(
            rows, cols, agent_rows, agent_cols, states
        )
        summary_writer.writerow(
            [
                0,
                int(s_counts.sum()),
                int(i_counts.sum()),
                int(r_counts.sum()),
                int(np.count_nonzero(occupancy)),
                int(occupancy.max(initial=0)),
                0,
            ]
        )
        write_location_rows(location_writer, 0, occupancy, s_counts, i_counts, r_counts)
        write_agent_rows(
            agent_writer,
            0,
            agent_rows,
            agent_cols,
            states,
            agent_rows,
            agent_cols,
            np.zeros(num_agents, dtype=bool),
            np.zeros(num_agents, dtype=np.int64),
            np.zeros(num_agents, dtype=np.float64),
            np.zeros(num_agents, dtype=bool),
            np.zeros(num_agents, dtype=bool),
        )

        for step in range(1, steps + 1):
            old_rows, old_cols, moved = move_agents(
                agent_rows, agent_cols, rows, cols, move_prob, movement_neighborhood, rng
            )

            occupancy, s_counts, i_counts, r_counts = build_location_stats(
                rows, cols, agent_rows, agent_cols, states
            )

            exposure_grid = i_counts.copy()
            if infection_neighborhood == "same_plus_news":
                exposure_grid = exposure_grid + cardinal_neighbor_sum(i_counts)

            exposure = exposure_grid[agent_rows, agent_cols]
            infection_prob = np.zeros(num_agents, dtype=np.float64)
            susceptible_mask = states == SUSCEPTIBLE
            infected_mask = states == INFECTED

            infection_prob[susceptible_mask] = 1.0 - np.power(
                1.0 - beta, exposure[susceptible_mask]
            )
            became_infected = susceptible_mask & (rng.random(num_agents) < infection_prob)
            became_recovered = infected_mask & (rng.random(num_agents) < gamma)

            states[became_infected] = INFECTED
            states[became_recovered] = RECOVERED

            occupancy, s_counts, i_counts, r_counts = build_location_stats(
                rows, cols, agent_rows, agent_cols, states
            )

            summary_writer.writerow(
                [
                    step,
                    int(s_counts.sum()),
                    int(i_counts.sum()),
                    int(r_counts.sum()),
                    int(np.count_nonzero(occupancy)),
                    int(occupancy.max(initial=0)),
                    int(moved.sum()),
                ]
            )
            write_location_rows(location_writer, step, occupancy, s_counts, i_counts, r_counts)
            write_agent_rows(
                agent_writer,
                step,
                agent_rows,
                agent_cols,
                states,
                old_rows,
                old_cols,
                moved,
                exposure,
                infection_prob,
                became_infected,
                became_recovered,
            )

    final_s = int(np.count_nonzero(states == SUSCEPTIBLE))
    final_i = int(np.count_nonzero(states == INFECTED))
    final_r = int(np.count_nonzero(states == RECOVERED))
    runtime_seconds = time.perf_counter() - start_time
    peak_memory_mb_used = peak_rss_mb()

    run_config["runtime_seconds"] = round(runtime_seconds, 6)
    run_config["peak_memory_mb"] = (
        round(peak_memory_mb_used, 3) if peak_memory_mb_used is not None else None
    )
    run_config["total_output_size_bytes"] = total_output_size_bytes(
        [summary_csv, agent_csv, location_csv, metadata_json]
    )
    write_run_metadata(metadata_json, run_config)

    print("Moving-agent SIR (single process)")
    print("Execution: CPU only (non-distributed)")
    print(f"Grid: rows={rows}, cols={cols}, locations={rows * cols}")
    print(f"Agents: num_agents={num_agents}, infected0={infected0}")
    print(
        "Rules: "
        f"infection_neighborhood={infection_neighborhood}, "
        f"movement_neighborhood={movement_neighborhood}, move_prob={move_prob}"
    )
    print(
        f"Parameters: beta={beta}, gamma={gamma}, steps={steps}, seed={seed} "
        "(step is a discrete simulation timestep; interpret as days if one step = one day)"
    )
    print(f"Final: S={final_s} I={final_i} R={final_r}")
    print(f"Runtime: {runtime_seconds:.3f} s")
    if peak_memory_mb_used is not None:
        print(f"Peak memory: {peak_memory_mb_used:.3f} MB")
    print(
        "Total output size: "
        f"{human_readable_size_from_bytes(run_config['total_output_size_bytes'])}"
    )
    print("Saved outputs:")
    print(f"  {summary_csv} ({human_readable_size(summary_csv)})")
    print(f"  {agent_csv} ({human_readable_size(agent_csv)})")
    print(f"  {location_csv} ({human_readable_size(location_csv)})")
    print(f"  {metadata_json} ({human_readable_size(metadata_json)})")
    if plot_summary_png:
        try:
            saved_plot = plot_summary(
                summary_csv,
                out_png=plot_out,
                title="Python SIR ABM Results",
                x_label="Day",
                config_json=metadata_json,
            )
        except RuntimeError as exc:
            print(f"plot skipped: {exc}")
            print("re-run with matplotlib available to generate the PNG summary plot")
        else:
            run_config["plot_out"] = str(saved_plot)
            run_config["total_output_size_bytes"] = total_output_size_bytes(
                [summary_csv, agent_csv, location_csv, metadata_json, saved_plot]
            )
            write_run_metadata(metadata_json, run_config)
            saved_plot = plot_summary(
                summary_csv,
                out_png=saved_plot,
                title="Python SIR ABM Results",
                x_label="Day",
                config_json=metadata_json,
            )
            print(f"  {saved_plot} ({human_readable_size(saved_plot)})")


def main():
    parser = argparse.ArgumentParser(
        description="CPU-only moving-agent SIR on a 2D grid with multiple agents per location"
    )
    parser.add_argument("--rows", type=int, default=100, help="Grid rows")
    parser.add_argument("--cols", type=int, default=100, help="Grid cols")
    parser.add_argument("--num-agents", type=int, default=10000, help="Total number of agents")
    parser.add_argument("--infected0", type=int, default=10, help="Initial infected agents")
    parser.add_argument("--beta", type=float, default=0.3, help="Infection intensity")
    parser.add_argument("--gamma", type=float, default=0.05, help="Recovery probability")
    parser.add_argument("--steps", type=int, default=30, help="Number of simulated steps")
    parser.add_argument(
        "--move-prob",
        type=float,
        default=0.50,
        help="Probability that an agent attempts movement at a step",
    )
    parser.add_argument(
        "--infection-neighborhood",
        type=str,
        default="same_plus_news",
        choices=["same_cell", "same_plus_news"],
        help="Exposure scope: same cell only, or same cell plus N/E/W/S locations",
    )
    parser.add_argument(
        "--movement-neighborhood",
        type=str,
        default="stay_news",
        choices=["news", "stay_news"],
        help="Possible movement directions when an agent attempts movement",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--out",
        type=str,
        default="sir_moving_agents",
        help="Output prefix for legacy flat-file logs and metadata",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional output directory for a self-contained run folder",
    )
    parser.add_argument(
        "--plot-summary",
        action="store_true",
        help="Also generate a PNG S/I/R summary plot after the run",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help="Optional PNG path for the summary plot; default derives it from the summary CSV name",
    )
    args = parser.parse_args()

    run_simulation(
        rows=args.rows,
        cols=args.cols,
        num_agents=args.num_agents,
        infected0=args.infected0,
        beta=args.beta,
        gamma=args.gamma,
        steps=args.steps,
        move_prob=args.move_prob,
        infection_neighborhood=args.infection_neighborhood,
        movement_neighborhood=args.movement_neighborhood,
        seed=args.seed,
        out_prefix=args.out,
        run_dir=args.run_dir,
        plot_summary_png=args.plot_summary,
        plot_out=args.plot_out,
    )


if __name__ == "__main__":
    main()
