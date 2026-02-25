import argparse

import numpy as np
from mpi4py import MPI


SUSCEPTIBLE = np.int8(0)
INFECTED = np.int8(1)
RECOVERED = np.int8(2)


def build_cartesian_comm():
    base = MPI.COMM_WORLD
    size = base.Get_size()
    dims = MPI.Compute_dims(size, [0, 0])
    # No periodic wraparound: physical edges have fewer neighbors.
    comm = base.Create_cart(dims=dims, periods=[False, False], reorder=True)
    north, south = comm.Shift(0, 1)
    west, east = comm.Shift(1, 1)
    return comm, dims, north, south, west, east


def split_axis(global_n, coord, dim_n):
    base = global_n // dim_n
    rem = global_n % dim_n
    local_n = base + (1 if coord < rem else 0)
    start = coord * base + min(coord, rem)
    return start, local_n


def exchange_halos(grid, comm, north, south, west, east):
    """
    Exchange only boundary strips with rank neighbors in a 2D Cartesian topology.
    grid shape: (local_rows + 2, local_cols + 2), halo-padded.
    """
    # Rows (contiguous)
    send_top = np.ascontiguousarray(grid[1, 1:-1])
    send_bottom = np.ascontiguousarray(grid[-2, 1:-1])
    recv_top = np.empty_like(send_top)
    recv_bottom = np.empty_like(send_bottom)

    comm.Sendrecv(sendbuf=send_top, dest=north, sendtag=10, recvbuf=recv_bottom, source=south, recvtag=10)
    comm.Sendrecv(sendbuf=send_bottom, dest=south, sendtag=11, recvbuf=recv_top, source=north, recvtag=11)

    if north != MPI.PROC_NULL:
        grid[0, 1:-1] = recv_top
    else:
        grid[0, 1:-1] = RECOVERED  # boundary treated as non-infectious
    if south != MPI.PROC_NULL:
        grid[-1, 1:-1] = recv_bottom
    else:
        grid[-1, 1:-1] = RECOVERED

    # Columns (need contiguous temp buffers)
    send_left = np.ascontiguousarray(grid[1:-1, 1])
    send_right = np.ascontiguousarray(grid[1:-1, -2])
    recv_left = np.empty_like(send_left)
    recv_right = np.empty_like(send_right)

    comm.Sendrecv(sendbuf=send_left, dest=west, sendtag=20, recvbuf=recv_right, source=east, recvtag=20)
    comm.Sendrecv(sendbuf=send_right, dest=east, sendtag=21, recvbuf=recv_left, source=west, recvtag=21)

    if west != MPI.PROC_NULL:
        grid[1:-1, 0] = recv_left
    else:
        grid[1:-1, 0] = RECOVERED
    if east != MPI.PROC_NULL:
        grid[1:-1, -1] = recv_right
    else:
        grid[1:-1, -1] = RECOVERED


def run_spatial_sir(
    global_rows=2048,
    global_cols=2048,
    steps=200,
    beta=0.25,
    gamma=0.05,
    infected_frac0=0.001,
    seed=0,
    output_prefix="sir_neighbor",
):
    comm, dims, north, south, west, east = build_cartesian_comm()
    rank = comm.Get_rank()
    coords = comm.Get_coords(rank)

    _, local_rows = split_axis(global_rows, coords[0], dims[0])
    _, local_cols = split_axis(global_cols, coords[1], dims[1])

    rng = np.random.default_rng(seed + rank)

    # Local state with 1-cell halo border.
    grid = np.full((local_rows + 2, local_cols + 2), SUSCEPTIBLE, dtype=np.int8)
    interior = grid[1:-1, 1:-1]

    # Initial infection as Bernoulli sample on local subdomain.
    interior[rng.random((local_rows, local_cols)) < infected_frac0] = INFECTED

    s_hist, i_hist, r_hist = [], [], []

    for _ in range(steps + 1):
        # Local-only communication: each rank talks only to N/S/E/W neighbors.
        exchange_halos(grid, comm, north, south, west, east)

        c = grid[1:-1, 1:-1]
        n = grid[0:-2, 1:-1]
        s = grid[2:, 1:-1]
        w = grid[1:-1, 0:-2]
        e = grid[1:-1, 2:]

        infected_neighbors = (
            (n == INFECTED).astype(np.int8)
            + (s == INFECTED).astype(np.int8)
            + (w == INFECTED).astype(np.int8)
            + (e == INFECTED).astype(np.int8)
        )

        susceptible_mask = c == SUSCEPTIBLE
        infected_mask = c == INFECTED

        # If k infected neighbors exist, infection probability is 1 - (1-beta)^k.
        p_infect = 1.0 - np.power(1.0 - beta, infected_neighbors)
        new_infected = susceptible_mask & (rng.random(c.shape) < p_infect)
        new_recovered = infected_mask & (rng.random(c.shape) < gamma)

        next_c = c.copy()
        next_c[new_infected] = INFECTED
        next_c[new_recovered] = RECOVERED
        grid[1:-1, 1:-1] = next_c

        local_s = np.int64(np.count_nonzero(next_c == SUSCEPTIBLE))
        local_i = np.int64(np.count_nonzero(next_c == INFECTED))
        local_r = np.int64(np.count_nonzero(next_c == RECOVERED))

        global_s = comm.allreduce(local_s, op=MPI.SUM)
        global_i = comm.allreduce(local_i, op=MPI.SUM)
        global_r = comm.allreduce(local_r, op=MPI.SUM)

        if rank == 0:
            s_hist.append(int(global_s))
            i_hist.append(int(global_i))
            r_hist.append(int(global_r))

    if rank == 0:
        peak_i = max(i_hist)
        peak_t = i_hist.index(peak_i)
        print(f"Cartesian process grid: {dims[0]} x {dims[1]} (total ranks={comm.Get_size()})")
        print(f"Final: S={s_hist[-1]} I={i_hist[-1]} R={r_hist[-1]}")
        print(f"Peak infected: {peak_i} at step {peak_t}")

        arr = np.column_stack((np.arange(len(s_hist)), s_hist, i_hist, r_hist))
        np.savetxt(
            f"{output_prefix}.csv",
            arr,
            delimiter=",",
            header="step,susceptible,infected,recovered",
            comments="",
            fmt="%d",
        )
        print(f"saved {output_prefix}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Spatial neighbor-based SIR with MPI domain decomposition"
    )
    parser.add_argument("--rows", type=int, default=2048, help="Global grid rows")
    parser.add_argument("--cols", type=int, default=2048, help="Global grid cols")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    parser.add_argument("--beta", type=float, default=0.25, help="Infection intensity")
    parser.add_argument("--gamma", type=float, default=0.05, help="Recovery probability")
    parser.add_argument("--infected-frac0", type=float, default=0.001, help="Initial infected fraction")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--out", type=str, default="sir_neighbor", help="Output prefix for CSV")
    args = parser.parse_args()

    run_spatial_sir(
        global_rows=args.rows,
        global_cols=args.cols,
        steps=args.steps,
        beta=args.beta,
        gamma=args.gamma,
        infected_frac0=args.infected_frac0,
        seed=args.seed,
        output_prefix=args.out,
    )


if __name__ == "__main__":
    main()
