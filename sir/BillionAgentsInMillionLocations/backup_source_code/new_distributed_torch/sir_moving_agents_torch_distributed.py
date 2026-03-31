import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:
    resource = None

try:
    import torch
except ImportError:
    torch = None

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from moving_sir_plotting import plot_summary


SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

STATE_NAME = {
    SUSCEPTIBLE: "S",
    INFECTED: "I",
    RECOVERED: "R",
}


def pick_device():
    if torch is None:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _env_first(names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _env_int(names, default=0):
    value = _env_first(names, None)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def distributed_env(comm):
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    local_rank = _env_int(
        ["LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "PMI_LOCAL_RANK"],
        0,
    )
    return rank, world_size, local_rank


def configure_device(base_device, local_rank):
    if base_device is None:
        return None
    if base_device.type == "cuda":
        if hasattr(torch.cuda, "set_device"):
            torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if base_device.type == "xpu":
        if hasattr(torch.xpu, "set_device"):
            torch.xpu.set_device(local_rank)
        return torch.device(f"xpu:{local_rank}")
    return base_device


def init_distributed():
    if torch is None:
        raise RuntimeError("This script requires PyTorch in the current Python environment.")
    if MPI is None:
        raise RuntimeError("This script requires mpi4py in the current Python environment.")
    comm = MPI.COMM_WORLD
    rank, world_size, local_rank = distributed_env(comm)
    return comm, rank, world_size, local_rank


def configured_backend():
    return "mpi4py"


def total_node_count(comm):
    if MPI is None:
        return 1
    node_name = MPI.Get_processor_name()
    if comm.Get_size() == 1:
        return 1
    return len(set(comm.allgather(node_name)))


def total_gpu_count_used(world_size, device):
    return int(world_size) if is_gpu_device(device) else 0


def cleanup_distributed(comm):
    comm.Barrier()


def is_distributed(world_size):
    return world_size > 1


def rank_print(rank, text):
    if rank == 0:
        print(text)


def movement_deltas(name):
    if name == "news":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if name == "stay_news":
        return [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    raise ValueError(f"Unsupported movement neighborhood: {name}")


def split_rows(global_rows, rank, world_size):
    base = global_rows // world_size
    rem = global_rows % world_size
    local_rows = base + (1 if rank < rem else 0)
    start = rank * base + min(rank, rem)
    return start, local_rows


def split_integer_by_weights(total, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if total < 0:
        raise ValueError("total must be non-negative")
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if weights.size == 0:
        return np.zeros(0, dtype=np.int64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        out = np.zeros(weights.shape[0], dtype=np.int64)
        out[: min(total, out.shape[0])] = 1
        return out

    exact = (weights / weight_sum) * float(total)
    out = np.floor(exact).astype(np.int64)
    remainder = int(total - int(out.sum()))
    if remainder > 0:
        fractional = exact - out
        order = np.argsort(-fractional, kind="mergesort")
        out[order[:remainder]] += 1
    return out


def build_initial_local_population(rows, cols, num_agents, infected0, seed, rank, world_size, row0, local_rows):
    local_row_counts = np.asarray([split_rows(rows, r, world_size)[1] for r in range(world_size)], dtype=np.int64)
    local_location_counts = local_row_counts * int(cols)
    local_agent_counts = split_integer_by_weights(num_agents, local_location_counts)
    local_agent_count = int(local_agent_counts[rank])

    agent_offsets = np.zeros(world_size, dtype=np.int64)
    if world_size > 1:
        agent_offsets[1:] = np.cumsum(local_agent_counts[:-1], dtype=np.int64)
    local_agent_offset = int(agent_offsets[rank])

    local_infected_counts = split_integer_by_weights(min(infected0, num_agents), local_agent_counts)
    local_infected_count = int(local_infected_counts[rank])

    rng = np.random.default_rng(seed + 1000003 * (rank + 1))
    if local_agent_count > 0 and local_rows > 0:
        agent_ids_np = np.arange(local_agent_offset, local_agent_offset + local_agent_count, dtype=np.int64)
        agent_rows_np = rng.integers(0, local_rows, size=local_agent_count, dtype=np.int64)
        agent_cols_np = rng.integers(0, cols, size=local_agent_count, dtype=np.int64)
        states_np = np.full(local_agent_count, SUSCEPTIBLE, dtype=np.int8)
        if local_infected_count > 0:
            infected_local_idx = rng.choice(local_agent_count, size=local_infected_count, replace=False)
            states_np[infected_local_idx] = INFECTED
    else:
        agent_ids_np = np.empty(0, dtype=np.int64)
        agent_rows_np = np.empty(0, dtype=np.int64)
        agent_cols_np = np.empty(0, dtype=np.int64)
        states_np = np.empty(0, dtype=np.int8)

    return agent_ids_np, agent_rows_np, agent_cols_np, states_np


def human_readable_size_from_bytes(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def human_readable_size(path):
    return human_readable_size_from_bytes(Path(path).stat().st_size)


def peak_rss_mb():
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def is_gpu_device(device):
    return device is not None and device.type in {"cuda", "xpu", "mps"}


def synchronize_device(device):
    if device is None:
        return
    if device.type == "cuda" and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
        torch.xpu.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def reset_peak_gpu_memory(device):
    if device is None:
        return
    if device.type == "cuda" and hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats(device)


def peak_gpu_memory_mb(device):
    if device is None:
        return None
    if device.type == "cuda" and hasattr(torch.cuda, "max_memory_allocated"):
        return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "max_memory_allocated"):
        return torch.xpu.max_memory_allocated(device) / (1024.0 * 1024.0)
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
    return None


def start_gpu_timer(device):
    if is_gpu_device(device):
        synchronize_device(device)
        return time.perf_counter()
    return None


def stop_gpu_timer(device, started_at):
    if started_at is None:
        return 0.0
    synchronize_device(device)
    return time.perf_counter() - started_at


def start_comm_timer():
    return time.perf_counter()


def stop_comm_timer(started_at):
    return time.perf_counter() - started_at


def write_run_metadata(path, config):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def resolve_output_paths(out_prefix, rank, run_dir=None):
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        agent_dir = run_dir / "agent_history"
        location_dir = run_dir / "location_history"
        agent_dir.mkdir(parents=True, exist_ok=True)
        location_dir.mkdir(parents=True, exist_ok=True)
        return (
            run_dir / "summary.csv",
            agent_dir / f"rank{rank}_agent_history.csv",
            location_dir / f"rank{rank}_location_history.csv",
            run_dir / "run_config.json",
        )

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    return (
        out_prefix.with_name(out_prefix.name + "_summary.csv"),
        out_prefix.with_name(out_prefix.name + f"_rank{rank}_agent_history.csv"),
        out_prefix.with_name(out_prefix.name + f"_rank{rank}_location_history.csv"),
        out_prefix.with_name(out_prefix.name + "_run_config.json"),
    )


def build_location_stats(local_rows, cols, agent_rows, agent_cols, states):
    n_cells = local_rows * cols
    if agent_rows.numel() == 0:
        zeros = torch.zeros((local_rows, cols), dtype=torch.int64, device=agent_rows.device)
        return zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()

    invalid_mask = (agent_rows < 0) | (agent_rows >= local_rows) | (agent_cols < 0) | (agent_cols >= cols)
    if bool(invalid_mask.any().item()):
        invalid_indices = torch.nonzero(invalid_mask, as_tuple=False).flatten()[:5]
        samples = [
            f"(row={int(agent_rows[idx].item())}, col={int(agent_cols[idx].item())})"
            for idx in invalid_indices.tolist()
        ]
        raise ValueError(
            "Invalid local agent coordinates detected before location statistics build: "
            f"{int(invalid_mask.sum().item())} invalid agents out of {int(agent_rows.numel())}. "
            f"Examples: {', '.join(samples)}"
        )

    cell_ids = agent_rows * cols + agent_cols
    occupancy = torch.bincount(cell_ids, minlength=n_cells).to(torch.int64)
    s_counts = torch.bincount(
        cell_ids,
        weights=(states == SUSCEPTIBLE).to(torch.float32),
        minlength=n_cells,
    ).to(torch.int64)
    i_counts = torch.bincount(
        cell_ids,
        weights=(states == INFECTED).to(torch.float32),
        minlength=n_cells,
    ).to(torch.int64)
    r_counts = torch.bincount(
        cell_ids,
        weights=(states == RECOVERED).to(torch.float32),
        minlength=n_cells,
    ).to(torch.int64)
    return (
        occupancy.reshape(local_rows, cols),
        s_counts.reshape(local_rows, cols),
        i_counts.reshape(local_rows, cols),
        r_counts.reshape(local_rows, cols),
    )


def cardinal_neighbor_sum_local(grid):
    out = torch.zeros_like(grid, dtype=torch.int64)
    out[:, 1:] += grid[:, :-1]
    out[:, :-1] += grid[:, 1:]
    return out


def exchange_vertical_halo_rows(i_counts, comm, rank, world_size, cols, device):
    if world_size == 1:
        top = torch.zeros(cols, dtype=torch.int64, device=device)
        bottom = torch.zeros(cols, dtype=torch.int64, device=device)
        return top, bottom, 0.0

    top_recv = np.zeros(cols, dtype=np.int64)
    bottom_recv = np.zeros(cols, dtype=np.int64)
    communication_seconds = 0.0
    if rank > 0:
        send_top = i_counts[0, :].detach().to("cpu", dtype=torch.int64).contiguous().numpy()
        comm_t0 = start_comm_timer()
        comm.Sendrecv(sendbuf=send_top, dest=rank - 1, sendtag=10, recvbuf=top_recv, source=rank - 1, recvtag=11)
        communication_seconds += stop_comm_timer(comm_t0)
    if rank < world_size - 1:
        send_bottom = i_counts[-1, :].detach().to("cpu", dtype=torch.int64).contiguous().numpy()
        comm_t0 = start_comm_timer()
        comm.Sendrecv(sendbuf=send_bottom, dest=rank + 1, sendtag=11, recvbuf=bottom_recv, source=rank + 1, recvtag=10)
        communication_seconds += stop_comm_timer(comm_t0)

    return (
        torch.as_tensor(top_recv, dtype=torch.int64, device=device),
        torch.as_tensor(bottom_recv, dtype=torch.int64, device=device),
        communication_seconds,
    )


def move_agents(agent_rows, agent_cols, local_rows, cols, move_prob, movement_rule, generator, device, rank, world_size):
    old_rows = agent_rows.clone()
    old_cols = agent_cols.clone()
    moved = torch.zeros(agent_rows.shape[0], dtype=torch.bool, device=device)

    if agent_rows.numel() == 0:
        return old_rows, old_cols, moved

    deltas = movement_deltas(movement_rule)
    want_move = torch.rand(agent_rows.shape[0], generator=generator, device=device) < move_prob
    min_row = -1 if rank > 0 else 0
    max_row = local_rows if rank < world_size - 1 else local_rows - 1
    for idx in torch.nonzero(want_move, as_tuple=False).flatten().tolist():
        valid = []
        current_row = int(agent_rows[idx].item())
        current_col = int(agent_cols[idx].item())
        for dr, dc in deltas:
            nr = current_row + dr
            nc = current_col + dc
            if 0 <= nc < cols and min_row <= nr <= max_row:
                valid.append((dr, dc))
        if not valid:
            continue
        choice = int(torch.randint(len(valid), (1,), generator=generator, device=device).item())
        dr, dc = valid[choice]
        agent_rows[idx] = current_row + dr
        agent_cols[idx] = current_col + dc
        moved[idx] = (dr != 0) or (dc != 0)

    return old_rows, old_cols, moved


def migrate_agents(comm, rank, world_size, row0, local_rows, agent_ids, agent_rows, agent_cols, states, old_rows, old_cols, moved, device):
    ids_np = agent_ids.detach().to("cpu").numpy()
    rows_np = agent_rows.detach().to("cpu").numpy()
    cols_np = agent_cols.detach().to("cpu").numpy()
    states_np = states.detach().to("cpu").numpy()
    old_rows_np = old_rows.detach().to("cpu").numpy()
    old_cols_np = old_cols.detach().to("cpu").numpy()
    moved_np = moved.detach().to(torch.int8).to("cpu").numpy()

    keep_mask = np.ones(ids_np.shape[0], dtype=bool)
    outgoing = []

    for idx in range(ids_np.shape[0]):
        local_row = int(rows_np[idx])
        dest_rank = None
        global_row = row0 + local_row
        if local_row < 0 and rank > 0:
            dest_rank = rank - 1
        elif local_row >= local_rows and rank < world_size - 1:
            dest_rank = rank + 1

        if dest_rank is None:
            continue

        keep_mask[idx] = False
        outgoing.append(
            (
                dest_rank,
                int(ids_np[idx]),
                int(global_row),
                int(cols_np[idx]),
                int(states_np[idx]),
                int(old_rows_np[idx]),
                int(old_cols_np[idx]),
                int(moved_np[idx]),
            )
        )

    ids_np = ids_np[keep_mask]
    rows_np = rows_np[keep_mask]
    cols_np = cols_np[keep_mask]
    states_np = states_np[keep_mask]
    old_rows_np = old_rows_np[keep_mask]
    old_cols_np = old_cols_np[keep_mask]
    moved_np = moved_np[keep_mask]

    if world_size > 1:
        comm_t0 = start_comm_timer()
        gathered = comm.allgather(outgoing)
        communication_seconds = stop_comm_timer(comm_t0)
        incoming = []
        for source_payload in gathered:
            for record in source_payload or []:
                if record[0] == rank:
                    incoming.append(record)
    else:
        incoming = []
        communication_seconds = 0.0

    if incoming:
        appended_ids = []
        appended_rows = []
        appended_cols = []
        appended_states = []
        appended_old_rows = []
        appended_old_cols = []
        appended_moved = []
        for _, agent_id, moved_row, moved_col, state, old_global_row, old_global_col, moved_flag in incoming:
            appended_ids.append(agent_id)
            appended_rows.append(moved_row - row0)
            appended_cols.append(moved_col)
            appended_states.append(state)
            appended_old_rows.append(old_global_row)
            appended_old_cols.append(old_global_col)
            appended_moved.append(moved_flag)

        ids_np = np.concatenate([ids_np, np.asarray(appended_ids, dtype=np.int64)])
        rows_np = np.concatenate([rows_np, np.asarray(appended_rows, dtype=np.int64)])
        cols_np = np.concatenate([cols_np, np.asarray(appended_cols, dtype=np.int64)])
        states_np = np.concatenate([states_np, np.asarray(appended_states, dtype=np.int8)])
        old_rows_np = np.concatenate([old_rows_np, np.asarray(appended_old_rows, dtype=np.int64)])
        old_cols_np = np.concatenate([old_cols_np, np.asarray(appended_old_cols, dtype=np.int64)])
        moved_np = np.concatenate([moved_np, np.asarray(appended_moved, dtype=np.int8)])

    return (
        torch.as_tensor(ids_np, dtype=torch.int64, device=device),
        torch.as_tensor(rows_np, dtype=torch.int64, device=device),
        torch.as_tensor(cols_np, dtype=torch.int64, device=device),
        torch.as_tensor(states_np, dtype=torch.int8, device=device),
        torch.as_tensor(old_rows_np, dtype=torch.int64, device=device),
        torch.as_tensor(old_cols_np, dtype=torch.int64, device=device),
        torch.as_tensor(moved_np.astype(np.bool_), dtype=torch.bool, device=device),
        communication_seconds,
    )


def total_output_size_bytes(paths):
    total = 0
    for path in paths:
        if path is None:
            continue
        candidate = Path(path)
        if candidate.exists():
            total += candidate.stat().st_size
    return total


def build_rank_output_list(run_dir, summary_csv, metadata_json, plot_out):
    paths = [summary_csv, metadata_json]
    run_dir = Path(run_dir)
    if run_dir.exists():
        agent_dir = run_dir / "agent_history"
        location_dir = run_dir / "location_history"
        if agent_dir.exists():
            paths.extend(sorted(str(path) for path in agent_dir.glob("rank*_agent_history.csv")))
        if location_dir.exists():
            paths.extend(sorted(str(path) for path in location_dir.glob("rank*_location_history.csv")))
    if plot_out is not None:
        paths.append(plot_out)
    return paths


def summarize_matching_files(directory, pattern):
    directory = Path(directory)
    if not directory.exists():
        return 0, 0
    matches = sorted(directory.glob(pattern))
    total_bytes = sum(path.stat().st_size for path in matches if path.exists())
    return len(matches), total_bytes


def write_local_location_rows(writer, step, rank, row0, occupancy, s_counts, i_counts, r_counts):
    occ_np = occupancy.detach().to("cpu").numpy()
    s_np = s_counts.detach().to("cpu").numpy()
    i_np = i_counts.detach().to("cpu").numpy()
    r_np = r_counts.detach().to("cpu").numpy()
    occupied = np.argwhere(occ_np > 0)
    for local_row, col in occupied:
        writer.writerow(
            [
                step,
                rank,
                int(row0 + local_row),
                int(col),
                int(occ_np[local_row, col]),
                int(s_np[local_row, col]),
                int(i_np[local_row, col]),
                int(r_np[local_row, col]),
            ]
        )


def write_local_agent_rows(
    writer,
    step,
    rank,
    row0,
    agent_ids,
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
    ids_np = agent_ids.detach().to("cpu").numpy()
    rows_np = agent_rows.detach().to("cpu").numpy()
    cols_np = agent_cols.detach().to("cpu").numpy()
    states_np = states.detach().to("cpu").numpy()
    old_rows_np = old_rows.detach().to("cpu").numpy()
    old_cols_np = old_cols.detach().to("cpu").numpy()
    moved_np = moved.detach().to(torch.int8).to("cpu").numpy()
    exposure_np = exposure.detach().to("cpu").numpy()
    infection_prob_np = infection_prob.detach().to("cpu").numpy()
    became_infected_np = became_infected.detach().to(torch.int8).to("cpu").numpy()
    became_recovered_np = became_recovered.detach().to(torch.int8).to("cpu").numpy()

    for idx in range(ids_np.shape[0]):
        writer.writerow(
            [
                step,
                rank,
                int(ids_np[idx]),
                int(row0 + rows_np[idx]),
                int(cols_np[idx]),
                STATE_NAME[int(states_np[idx])],
                int(old_rows_np[idx]),
                int(old_cols_np[idx]),
                int(row0 + rows_np[idx]),
                int(cols_np[idx]),
                int(moved_np[idx]),
                int(exposure_np[idx]),
                float(infection_prob_np[idx]),
                int(became_infected_np[idx]),
                int(became_recovered_np[idx]),
            ]
        )


def gather_global_counts(comm, world_size, local_counts):
    local_counts = local_counts.to("cpu", dtype=torch.int64)
    if is_distributed(world_size):
        sendbuf = local_counts.numpy()
        recvbuf = np.zeros_like(sendbuf)
        comm_t0 = start_comm_timer()
        comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
        return torch.as_tensor(recvbuf, dtype=torch.int64), stop_comm_timer(comm_t0)
    return local_counts, 0.0


def gather_global_max(comm, world_size, local_value):
    if is_distributed(world_size):
        comm_t0 = start_comm_timer()
        return int(comm.allreduce(int(local_value), op=MPI.MAX)), stop_comm_timer(comm_t0)
    return int(local_value), 0.0


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
    out_prefix="sir_moving_agents_torch_distributed",
    run_dir=None,
    plot_summary_png=False,
    plot_out=None,
):
    if torch is None:
        raise RuntimeError("This script requires PyTorch in the current Python environment.")
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

    comm, rank, world_size, local_rank = init_distributed()
    base_device = pick_device()
    device = configure_device(base_device, local_rank)
    job_total_nodes = total_node_count(comm)
    job_total_mpi_ranks = int(world_size)
    job_total_gpus_used = total_gpu_count_used(world_size, device)
    start_time = time.perf_counter()
    output_io_seconds_local = 0.0
    gpu_compute_seconds_local = 0.0
    communication_seconds_local = 0.0
    plot_generation_seconds = None
    if is_gpu_device(device):
        reset_peak_gpu_memory(device)

    row0, local_rows = split_rows(rows, rank, world_size)
    summary_csv, agent_csv, location_csv, metadata_json = resolve_output_paths(out_prefix, rank, run_dir=run_dir)

    agent_ids_np, agent_rows_np, agent_cols_np, states_np = build_initial_local_population(
        rows=rows,
        cols=cols,
        num_agents=num_agents,
        infected0=infected0,
        seed=seed,
        rank=rank,
        world_size=world_size,
        row0=row0,
        local_rows=local_rows,
    )
    agent_ids = torch.as_tensor(agent_ids_np, dtype=torch.int64, device=device)
    agent_rows = torch.as_tensor(agent_rows_np, dtype=torch.int64, device=device)
    agent_cols = torch.as_tensor(agent_cols_np, dtype=torch.int64, device=device)
    states = torch.as_tensor(states_np, dtype=torch.int8, device=device)

    generator = torch.Generator(device=device if device.type != "mps" else "cpu")
    generator.manual_seed(seed + rank + 1)

    run_config = {
        "mode": "torch_distributed_row_partition",
        "execution": "distributed_torch_gpu",
        "backend": configured_backend(),
        "world_size": world_size,
        "total_nodes": job_total_nodes,
        "total_mpi_ranks": job_total_mpi_ranks,
        "total_gpus_used": job_total_gpus_used,
        "rank": rank,
        "device": str(device),
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
        "initialization": "local_rank_owned_generation",
        "run_dir": str(run_dir) if run_dir else None,
        "plot_summary_png": plot_summary_png,
        "plot_out": str(plot_out) if plot_out else None,
        "run_config_json": str(metadata_json),
        "summary_csv": str(summary_csv),
        "distributed_runtime_seconds": None,
        "total_wall_time_seconds": None,
        "peak_rank_memory_mb": None,
        "host_memory_per_rank_mb": None,
        "total_host_memory_mb": None,
        "total_gpu_compute_seconds": None,
        "gpu_compute_time_max_rank_seconds": None,
        "gpu_compute_fraction_of_wall": None,
        "peak_gpu_memory_mb": None,
        "gpu_memory_per_rank_mb": None,
        "total_gpu_memory_mb": None,
        "total_communication_seconds": None,
        "communication_time_max_rank_seconds": None,
        "total_output_io_seconds": None,
        "plot_generation_seconds": None,
        "total_output_size_bytes": None,
    }
    if rank == 0:
        io_t0 = time.perf_counter()
        write_run_metadata(metadata_json, run_config)
        output_io_seconds_local += time.perf_counter() - io_t0

    with open(agent_csv, "w", newline="", encoding="utf-8") as agent_f, open(location_csv, "w", newline="", encoding="utf-8") as location_f:
        agent_writer = csv.writer(agent_f)
        location_writer = csv.writer(location_f)
        agent_writer.writerow(["step", "rank", "agent_id", "row", "col", "state", "from_row", "from_col", "to_row", "to_col", "moved", "exposure", "infection_probability", "became_infected", "became_recovered"])
        location_writer.writerow(["step", "rank", "row", "col", "occupancy", "susceptible", "infected", "recovered"])

        if rank == 0:
            summary_f = open(summary_csv, "w", newline="", encoding="utf-8")
            summary_writer = csv.writer(summary_f)
            summary_writer.writerow(["step", "susceptible", "infected", "recovered", "occupied_locations", "max_occupancy", "moved_agents"])
        else:
            summary_f = None
            summary_writer = None

        gpu_t0 = start_gpu_timer(device)
        occupancy, s_counts, i_counts, r_counts = build_location_stats(local_rows, cols, agent_rows, agent_cols, states)
        gpu_compute_seconds_local += stop_gpu_timer(device, gpu_t0)
        io_t0 = time.perf_counter()
        write_local_location_rows(location_writer, 0, rank, row0, occupancy, s_counts, i_counts, r_counts)
        write_local_agent_rows(
            agent_writer,
            0,
            rank,
            row0,
            agent_ids,
            agent_rows,
            agent_cols,
            states,
            row0 + agent_rows,
            agent_cols,
            torch.zeros(agent_ids.shape[0], dtype=torch.bool, device=device),
            torch.zeros(agent_ids.shape[0], dtype=torch.int64, device=device),
            torch.zeros(agent_ids.shape[0], dtype=torch.float64, device=device),
            torch.zeros(agent_ids.shape[0], dtype=torch.bool, device=device),
            torch.zeros(agent_ids.shape[0], dtype=torch.bool, device=device),
        )
        output_io_seconds_local += time.perf_counter() - io_t0

        local_counts = torch.tensor(
            [
                int((states == SUSCEPTIBLE).sum().item()),
                int((states == INFECTED).sum().item()),
                int((states == RECOVERED).sum().item()),
                int((occupancy > 0).sum().item()),
                0,
            ],
            dtype=torch.int64,
        )
        global_counts, comm_seconds = gather_global_counts(comm, world_size, local_counts)
        communication_seconds_local += comm_seconds
        max_occ, comm_seconds = gather_global_max(comm, world_size, int(occupancy.max().item()) if occupancy.numel() > 0 else 0)
        communication_seconds_local += comm_seconds
        if rank == 0:
            io_t0 = time.perf_counter()
            summary_writer.writerow([0, int(global_counts[0].item()), int(global_counts[1].item()), int(global_counts[2].item()), int(global_counts[3].item()), int(max_occ), 0])
            output_io_seconds_local += time.perf_counter() - io_t0

        for step in range(1, steps + 1):
            gpu_t0 = start_gpu_timer(device)
            old_rows_local, old_cols, moved = move_agents(
                agent_rows,
                agent_cols,
                local_rows,
                cols,
                move_prob,
                movement_neighborhood,
                generator,
                device,
                rank,
                world_size,
            )
            old_rows_global = row0 + old_rows_local
            gpu_compute_seconds_local += stop_gpu_timer(device, gpu_t0)

            agent_ids, agent_rows, agent_cols, states, old_rows_global, old_cols, moved, comm_seconds = migrate_agents(
                comm,
                rank,
                world_size,
                row0,
                local_rows,
                agent_ids,
                agent_rows,
                agent_cols,
                states,
                old_rows_global,
                old_cols,
                moved,
                device,
            )
            communication_seconds_local += comm_seconds

            gpu_t0 = start_gpu_timer(device)
            occupancy, s_counts, i_counts, r_counts = build_location_stats(local_rows, cols, agent_rows, agent_cols, states)
            exposure_grid = i_counts.clone()
            if infection_neighborhood == "same_plus_news":
                exposure_grid = exposure_grid + cardinal_neighbor_sum_local(i_counts)
                top_halo, bottom_halo, comm_seconds = exchange_vertical_halo_rows(i_counts, comm, rank, world_size, cols, device)
                communication_seconds_local += comm_seconds
                exposure_grid[0, :] += top_halo
                exposure_grid[-1, :] += bottom_halo

            if agent_rows.numel() > 0:
                exposure = exposure_grid[agent_rows, agent_cols]
            else:
                exposure = torch.empty(0, dtype=torch.int64, device=device)

            infection_prob = torch.zeros(agent_ids.shape[0], dtype=torch.float64, device=device)
            susceptible_mask = states == SUSCEPTIBLE
            infected_mask = states == INFECTED
            if susceptible_mask.any():
                infection_prob[susceptible_mask] = 1.0 - torch.pow(
                    torch.tensor(1.0 - beta, dtype=torch.float64, device=device),
                    exposure[susceptible_mask].to(torch.float64),
                )

            rand_infect = torch.rand(agent_ids.shape[0], generator=generator, device=device, dtype=torch.float64)
            rand_recover = torch.rand(agent_ids.shape[0], generator=generator, device=device, dtype=torch.float64)
            became_infected = susceptible_mask & (rand_infect < infection_prob)
            became_recovered = infected_mask & (rand_recover < gamma)
            states[became_infected] = INFECTED
            states[became_recovered] = RECOVERED

            occupancy, s_counts, i_counts, r_counts = build_location_stats(local_rows, cols, agent_rows, agent_cols, states)
            gpu_compute_seconds_local += stop_gpu_timer(device, gpu_t0)
            io_t0 = time.perf_counter()
            write_local_location_rows(location_writer, step, rank, row0, occupancy, s_counts, i_counts, r_counts)
            write_local_agent_rows(
                agent_writer,
                step,
                rank,
                row0,
                agent_ids,
                agent_rows,
                agent_cols,
                states,
                old_rows_global,
                old_cols,
                moved,
                exposure,
                infection_prob,
                became_infected,
                became_recovered,
            )
            output_io_seconds_local += time.perf_counter() - io_t0

            local_counts = torch.tensor(
                [
                    int((states == SUSCEPTIBLE).sum().item()),
                    int((states == INFECTED).sum().item()),
                    int((states == RECOVERED).sum().item()),
                    int((occupancy > 0).sum().item()),
                    int(moved.sum().item()),
                ],
                dtype=torch.int64,
            )
            global_counts, comm_seconds = gather_global_counts(comm, world_size, local_counts)
            communication_seconds_local += comm_seconds
            max_occ, comm_seconds = gather_global_max(comm, world_size, int(occupancy.max().item()) if occupancy.numel() > 0 else 0)
            communication_seconds_local += comm_seconds
            if rank == 0:
                io_t0 = time.perf_counter()
                summary_writer.writerow(
                    [
                        step,
                        int(global_counts[0].item()),
                        int(global_counts[1].item()),
                        int(global_counts[2].item()),
                        int(global_counts[3].item()),
                        int(max_occ),
                        int(global_counts[4].item()),
                    ]
                )
                output_io_seconds_local += time.perf_counter() - io_t0

        if summary_f is not None:
            summary_f.close()

    if is_distributed(world_size):
        comm_t0 = start_comm_timer()
        comm.Barrier()
        communication_seconds_local += stop_comm_timer(comm_t0)

    local_final = torch.tensor(
        [
            int((states == SUSCEPTIBLE).sum().item()),
            int((states == INFECTED).sum().item()),
            int((states == RECOVERED).sum().item()),
        ],
        dtype=torch.int64,
    )
    global_final, comm_seconds = gather_global_counts(comm, world_size, local_final)
    communication_seconds_local += comm_seconds

    local_runtime = time.perf_counter() - start_time
    if is_distributed(world_size):
        comm_t0 = start_comm_timer()
        distributed_runtime_seconds = float(comm.allreduce(local_runtime, op=MPI.MAX))
        communication_seconds_local += stop_comm_timer(comm_t0)
    else:
        distributed_runtime_seconds = float(local_runtime)

    local_peak_memory = peak_rss_mb()
    if is_distributed(world_size):
        comm_t0 = start_comm_timer()
        peak_rank_memory_mb = float(
            comm.allreduce(0.0 if local_peak_memory is None else float(local_peak_memory), op=MPI.MAX)
        )
        communication_seconds_local += stop_comm_timer(comm_t0)
    else:
        peak_rank_memory_mb = float(0.0 if local_peak_memory is None else local_peak_memory)
    if is_distributed(world_size):
        comm_t0 = start_comm_timer()
        total_host_memory_mb = float(
            comm.allreduce(0.0 if local_peak_memory is None else float(local_peak_memory), op=MPI.SUM)
        )
        communication_seconds_local += stop_comm_timer(comm_t0)
    else:
        total_host_memory_mb = float(0.0 if local_peak_memory is None else local_peak_memory)
    local_peak_gpu_memory = peak_gpu_memory_mb(device)
    if is_distributed(world_size):
        comm_t0 = start_comm_timer()
        peak_gpu_memory_used_mb = float(
            comm.allreduce(0.0 if local_peak_gpu_memory is None else float(local_peak_gpu_memory), op=MPI.MAX)
        )
        communication_seconds_local += stop_comm_timer(comm_t0)
    else:
        peak_gpu_memory_used_mb = float(0.0 if local_peak_gpu_memory is None else local_peak_gpu_memory)
    if is_distributed(world_size):
        comm_t0 = start_comm_timer()
        total_gpu_memory_used_mb = float(
            comm.allreduce(0.0 if local_peak_gpu_memory is None else float(local_peak_gpu_memory), op=MPI.SUM)
        )
        communication_seconds_local += stop_comm_timer(comm_t0)
    else:
        total_gpu_memory_used_mb = float(0.0 if local_peak_gpu_memory is None else local_peak_gpu_memory)
    if is_distributed(world_size):
        comm_t0 = start_comm_timer()
        total_gpu_compute_seconds = float(comm.allreduce(float(gpu_compute_seconds_local), op=MPI.SUM))
        communication_seconds_local += stop_comm_timer(comm_t0)
        comm_t0 = start_comm_timer()
        gpu_compute_time_max_rank_seconds = float(comm.allreduce(float(gpu_compute_seconds_local), op=MPI.MAX))
        communication_seconds_local += stop_comm_timer(comm_t0)
    else:
        total_gpu_compute_seconds = float(gpu_compute_seconds_local)
        gpu_compute_time_max_rank_seconds = float(gpu_compute_seconds_local)
    if is_distributed(world_size):
        comm_t0 = start_comm_timer()
        total_output_io_seconds = float(comm.allreduce(float(output_io_seconds_local), op=MPI.SUM))
        communication_seconds_local += stop_comm_timer(comm_t0)
    else:
        total_output_io_seconds = float(output_io_seconds_local)
    if is_distributed(world_size):
        total_communication_seconds = float(comm.allreduce(float(communication_seconds_local), op=MPI.SUM))
        communication_time_max_rank_seconds = float(comm.allreduce(float(communication_seconds_local), op=MPI.MAX))
    else:
        total_communication_seconds = float(communication_seconds_local)
        communication_time_max_rank_seconds = float(communication_seconds_local)
    communication_fraction_of_wall = (
        (communication_time_max_rank_seconds / distributed_runtime_seconds) if distributed_runtime_seconds > 0.0 else None
    )
    gpu_compute_fraction_of_wall = (
        (gpu_compute_time_max_rank_seconds / distributed_runtime_seconds) if distributed_runtime_seconds > 0.0 else None
    )
    local_output_io_at_reduce = output_io_seconds_local

    if rank == 0:
        run_config["distributed_runtime_seconds"] = round(distributed_runtime_seconds, 6)
        run_config["total_wall_time_seconds"] = round(distributed_runtime_seconds, 6)
        run_config["peak_rank_memory_mb"] = round(peak_rank_memory_mb, 3)
        run_config["host_memory_per_rank_mb"] = round(peak_rank_memory_mb, 3)
        run_config["total_host_memory_mb"] = round(total_host_memory_mb, 3)
        run_config["total_gpu_compute_seconds"] = round(total_gpu_compute_seconds, 6)
        run_config["gpu_compute_time_max_rank_seconds"] = round(gpu_compute_time_max_rank_seconds, 6)
        run_config["gpu_compute_fraction_of_wall"] = (
            round(gpu_compute_fraction_of_wall, 6) if gpu_compute_fraction_of_wall is not None else None
        )
        run_config["peak_gpu_memory_mb"] = round(peak_gpu_memory_used_mb, 3) if is_gpu_device(device) else None
        run_config["gpu_memory_per_rank_mb"] = round(peak_gpu_memory_used_mb, 3) if is_gpu_device(device) else None
        run_config["total_gpu_memory_mb"] = round(total_gpu_memory_used_mb, 3) if is_gpu_device(device) else None
        run_config["total_communication_seconds"] = round(total_communication_seconds, 6)
        run_config["communication_time_max_rank_seconds"] = round(communication_time_max_rank_seconds, 6)
        run_config["communication_fraction_of_wall"] = (
            round(communication_fraction_of_wall, 6) if communication_fraction_of_wall is not None else None
        )
        run_config["total_output_io_seconds"] = round(total_output_io_seconds, 6)
        run_config["plot_generation_seconds"] = plot_generation_seconds
        run_config["total_output_size_bytes"] = total_output_size_bytes(
            build_rank_output_list(run_dir or Path(summary_csv).parent, summary_csv, metadata_json, None)
        )
        io_t0 = time.perf_counter()
        write_run_metadata(metadata_json, run_config)
        output_io_seconds_local += time.perf_counter() - io_t0

        print("Moving-agent SIR (PyTorch distributed with MPI communication)")
        print(f"Backend: {configured_backend()}")
        print(f"Total nodes: {job_total_nodes}")
        print(f"Total MPI ranks: {job_total_mpi_ranks}")
        print(f"Total GPUs used: {job_total_gpus_used}")
        print(f"Device: {device}")
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
        print(f"Final: S={int(global_final[0].item())} I={int(global_final[1].item())} R={int(global_final[2].item())}")
        print(f"Total wall time: {distributed_runtime_seconds:.3f} s")
        print(f"Host memory per rank (peak): {peak_rank_memory_mb:.3f} MB")
        print(f"Total host memory (all ranks): {total_host_memory_mb:.3f} MB")
        if is_gpu_device(device):
            if gpu_compute_fraction_of_wall is not None:
                print(
                    f"GPU compute time on slowest rank: {gpu_compute_time_max_rank_seconds:.3f} s "
                    f"({gpu_compute_fraction_of_wall * 100.0:.1f}% of wall time)"
                )
            else:
                print(f"GPU compute time on slowest rank: {gpu_compute_time_max_rank_seconds:.3f} s")
            print(f"GPU compute time (all ranks summed): {total_gpu_compute_seconds:.3f} s")
            print(f"GPU memory per rank (peak): {peak_gpu_memory_used_mb:.3f} MB")
            print(f"Total GPU memory (all ranks): {total_gpu_memory_used_mb:.3f} MB")
        if communication_fraction_of_wall is not None:
            print(
                f"Communication time on slowest rank: {communication_time_max_rank_seconds:.3f} s "
                f"({communication_fraction_of_wall * 100.0:.1f}% of wall time)"
            )
        else:
            print(f"Communication time on slowest rank: {communication_time_max_rank_seconds:.3f} s")
        print(f"Communication time (all ranks summed): {total_communication_seconds:.3f} s")
        print(f"Total output I/O time: {total_output_io_seconds:.3f} s")
        print(
            "Saved outputs:"
        )
        print(f"  {summary_csv} ({human_readable_size(summary_csv)})")
        print(f"  {metadata_json} ({human_readable_size(metadata_json)})")
        if run_dir is not None:
            agent_dir = Path(run_dir) / "agent_history"
            location_dir = Path(run_dir) / "location_history"
            agent_count, agent_bytes = summarize_matching_files(agent_dir, "rank*_agent_history.csv")
            location_count, location_bytes = summarize_matching_files(location_dir, "rank*_location_history.csv")
            print(
                f"  {agent_dir} ({agent_count} files, {human_readable_size_from_bytes(agent_bytes)})"
            )
            print(
                f"  {location_dir} ({location_count} files, {human_readable_size_from_bytes(location_bytes)})"
            )
        if plot_summary_png:
            try:
                plot_t0 = time.perf_counter()
                saved_plot = plot_summary(
                    summary_csv,
                    out_png=plot_out,
                    title="Distributed Moving-Agent SIR",
                    x_label="Day",
                    config_json=metadata_json,
                )
                plot_generation_seconds = time.perf_counter() - plot_t0
            except RuntimeError as exc:
                print(f"plot skipped: {exc}")
                print("re-run with matplotlib available to generate the PNG summary plot")
            else:
                run_config["plot_out"] = str(saved_plot)
                run_config["plot_generation_seconds"] = round(plot_generation_seconds, 6)
                run_config["total_output_size_bytes"] = total_output_size_bytes(
                    build_rank_output_list(run_dir or Path(summary_csv).parent, summary_csv, metadata_json, saved_plot)
                )
                io_t0 = time.perf_counter()
                write_run_metadata(metadata_json, run_config)
                output_io_seconds_local += time.perf_counter() - io_t0
                plot_t0 = time.perf_counter()
                saved_plot = plot_summary(
                    summary_csv,
                    out_png=saved_plot,
                    title="Distributed Moving-Agent SIR",
                    x_label="Day",
                    config_json=metadata_json,
                )
                plot_generation_seconds += time.perf_counter() - plot_t0
                run_config["plot_generation_seconds"] = round(plot_generation_seconds, 6)
                io_t0 = time.perf_counter()
                write_run_metadata(metadata_json, run_config)
                output_io_seconds_local += time.perf_counter() - io_t0
                extra_rank0_output_io = output_io_seconds_local - local_output_io_at_reduce
                run_config["total_output_io_seconds"] = round(total_output_io_seconds + extra_rank0_output_io, 6)
                io_t0 = time.perf_counter()
                write_run_metadata(metadata_json, run_config)
                output_io_seconds_local += time.perf_counter() - io_t0
                print(f"Plot generation time: {plot_generation_seconds:.3f} s")
                print(f"  {saved_plot} ({human_readable_size(saved_plot)})")

    cleanup_distributed(comm)


def main():
    parser = argparse.ArgumentParser(description="Distributed moving-agent SIR using PyTorch local tensors with mpi4py communication")
    parser.add_argument("--rows", type=int, default=100, help="Global grid rows")
    parser.add_argument("--cols", type=int, default=100, help="Global grid cols")
    parser.add_argument("--num-agents", type=int, default=10000, help="Total number of agents")
    parser.add_argument("--infected0", type=int, default=10, help="Initial infected agents")
    parser.add_argument("--beta", type=float, default=0.3, help="Infection intensity")
    parser.add_argument("--gamma", type=float, default=0.05, help="Recovery probability")
    parser.add_argument("--steps", type=int, default=30, help="Simulation timesteps; interpret as days if one step per day")
    parser.add_argument("--move-prob", type=float, default=0.50, help="Movement attempt probability")
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
    parser.add_argument("--out", type=str, default="sir_moving_agents_torch_distributed", help="Output prefix for flat-file logs and metadata")
    parser.add_argument("--run-dir", type=str, default=None, help="Optional output directory for a self-contained run folder")
    parser.add_argument("--plot-summary", action="store_true", help="Also generate a PNG summary plot on rank 0 after the run")
    parser.add_argument("--plot-out", type=str, default=None, help="Optional PNG path for the summary plot; default derives it from the summary CSV name")
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
