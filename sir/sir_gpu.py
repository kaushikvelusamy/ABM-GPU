import argparse
import csv
import os
import socket

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist


def pick_device(local_rank=0):
    # Priority order: CUDA (NVIDIA), MPS (Apple), XPU (Intel), CPU fallback.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        if hasattr(torch.xpu, "set_device"):
            torch.xpu.set_device(local_rank)
        return torch.device(f"xpu:{local_rank}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _env_int(name, default=0):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def setup_distributed_from_mpi(enabled=False, backend="xccl"):
    rank = 0
    world_size = 1
    local_rank = _env_int("MPI_LOCALRANKID", _env_int("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    is_distributed = False

    mpi_detected = any(
        key in os.environ
        for key in ["PMI_SIZE", "PMIX_RANK", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALRANKID"]
    )

    if enabled or mpi_detected:
        try:
            from mpi4py import MPI  # Imported lazily so single-rank mode has no mpi4py hard dependency.
        except ImportError as exc:
            raise RuntimeError(
                "MPI mode requested/detected but mpi4py is not available in this Python environment."
            ) from exc

        # Read rank topology from MPI launcher.
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        is_distributed = world_size > 1

        if is_distributed:
            # torch.distributed env:// expects these variables to be present.
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ.setdefault("MASTER_PORT", "29500")

            # Share one master hostname from rank 0 to all ranks.
            master_addr = comm.bcast(socket.gethostname() if rank == 0 else None, root=0)
            os.environ.setdefault("MASTER_ADDR", master_addr)

            if backend == "xccl":
                # Required for Intel oneCCL/XCCL backend in PyTorch.
                try:
                    import oneccl_bindings_for_pytorch  # noqa: F401
                except ImportError as exc:
                    raise RuntimeError(
                        "backend=xccl requires oneccl_bindings_for_pytorch in this environment."
                    ) from exc

            dist.init_process_group(backend=backend, init_method="env://")

    return is_distributed, rank, world_size, local_rank


def run_sir(
    population=10000,
    infected0=10,
    beta=0.30,
    gamma=0.10,
    steps=160,
    mpi=False,
    backend="xccl",
):
    """
    Simulate the classic SIR model in discrete time.

    S: susceptible people
    I: infected people
    R: recovered (or removed) people

    At each day:
    new_infections = beta * S * I / N
    new_recoveries = gamma * I
    """
    is_distributed, rank, world_size, local_rank = setup_distributed_from_mpi(
        enabled=mpi, backend=backend
    )
    device = pick_device(local_rank=local_rank)

    if rank == 0:
        print("Device:", device)
        if is_distributed:
            print(f"MPI run: world_size={world_size}, backend={backend}")

    # Split total population across ranks. First `remainder` ranks get +1 person.
    base = population // world_size
    remainder = population % world_size
    local_n_int = base + (1 if rank < remainder else 0)

    # Seed initial infections on rank 0; cap so local state is always valid.
    local_i0 = infected0 if rank == 0 else 0
    local_i0 = min(local_i0, local_n_int)
    local_s0 = local_n_int - local_i0

    # Torch tensors keep arithmetic on the selected device.
    s = torch.tensor(float(local_s0), device=device)
    i = torch.tensor(float(local_i0), device=device)
    r = torch.tensor(0.0, device=device)

    s_hist, i_hist, r_hist = [], [], []

    for _ in range(steps + 1):
        if is_distributed:
            # torch.distributed all_reduce gives global compartment totals over all ranks.
            state = torch.tensor([s.item(), i.item(), r.item(), float(local_n_int)], device=device)
            dist.all_reduce(state, op=dist.ReduceOp.SUM)
            global_s = float(state[0].item())
            global_i = float(state[1].item())
            global_r = float(state[2].item())
            global_n = float(state[3].item())
        else:
            global_s = float(s.item())
            global_i = float(i.item())
            global_r = float(r.item())
            global_n = float(local_n_int)

        if rank == 0:
            s_hist.append(global_s)
            i_hist.append(global_i)
            r_hist.append(global_r)

        # Well-mixed model: each local shard uses global infected fraction I/N.
        i_global_t = torch.tensor(global_i, device=device)
        n_global_t = torch.tensor(global_n, device=device)
        new_infections = beta * s * i_global_t / n_global_t
        new_recoveries = gamma * i

        # clamp avoids tiny negative values from floating-point drift.
        s = torch.clamp(s - new_infections, min=0.0)
        i = torch.clamp(i + new_infections - new_recoveries, min=0.0)
        r = torch.clamp(r + new_recoveries, min=0.0)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        peak_i = max(i_hist)
        peak_day = i_hist.index(peak_i)
        print(f"Final: S={s_hist[-1]:.1f} I={i_hist[-1]:.1f} R={r_hist[-1]:.1f}")
        print(f"Peak infected: {peak_i:.1f} at day {peak_day}")
        return s_hist, i_hist, r_hist
    return None, None, None


def plot_sir(s_hist, i_hist, r_hist, out_png="sir_curve.png"):
    """Plot S, I, and R curves versus day and save to a PNG file."""
    x = list(range(len(s_hist)))
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, s_hist, label="Susceptible")
    plt.plot(x, i_hist, label="Infected")
    plt.plot(x, r_hist, label="Recovered")
    plt.xlabel("Day")
    plt.ylabel("People")
    plt.title("Simple SIR model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"saved {out_png}")


def write_sir_csv(s_hist, i_hist, r_hist, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "susceptible", "infected", "recovered"])
        for step, (s, i, r) in enumerate(zip(s_hist, i_hist, r_hist)):
            writer.writerow([step, s, i, r])
    print(f"saved {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple SIR model (PyTorch compute + optional MPI scaling)"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=10000,
        help="Total population N (constant over the simulation)",
    )
    parser.add_argument(
        "--infected0",
        type=int,
        default=10,
        help="Initial infected count at day 0",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.30,
        help="Infection rate parameter (larger means faster spread)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.10,
        help="Recovery rate parameter (larger means faster recovery)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=160,
        help="Number of simulated days",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="sir_curve.png",
        help="Output PNG file path for the S/I/R plot",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional output CSV path; default uses <out>.csv",
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help="Enable MPI-launched distributed mode (mpiexec + torch.distributed)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="xccl",
        choices=["xccl", "nccl", "gloo"],
        help="torch.distributed backend (default: xccl)",
    )
    args = parser.parse_args()

    s_hist, i_hist, r_hist = run_sir(
        population=args.population,
        infected0=args.infected0,
        beta=args.beta,
        gamma=args.gamma,
        steps=args.steps,
        mpi=args.mpi,
        backend=args.backend,
    )
    if s_hist is not None:
        out_csv = args.out_csv
        if out_csv is None:
            out_csv = f"{os.path.splitext(args.out)[0]}.csv"
        write_sir_csv(s_hist, i_hist, r_hist, out_csv=out_csv)
        plot_sir(s_hist, i_hist, r_hist, out_png=args.out)
