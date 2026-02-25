Neighbor Selection in SIR: Both Codes

Files compared:
- sir_gpu.py
- sir_neighbor_mpi.py

1) Neighbor selection in sir_gpu.py (well-mixed model)

How neighbors are chosen:
- There are no explicit per-agent or per-cell neighbors.
- Every susceptible person is assumed to mix with the whole population.
- Infection pressure comes from the global infected fraction I/N.

Update idea:
- new_infections = beta * S * I / N
- new_recoveries = gamma * I

Distributed behavior:
- Each rank holds a shard of S, I, R.
- Ranks use collective reductions to compute global S, I, R, N.
- Each rank updates using the same global I/N value.

Interpretation:
- "Neighbor" means "everyone in population" (mean-field assumption).


2) Neighbor selection in sir_neighbor_mpi.py (spatial neighbor model)

How neighbors are chosen:
- Population is arranged on a 2D grid.
- Each cell interacts only with local geometric neighbors: North, South, East, West.
- Border cells need off-rank neighbor values when decomposition crosses rank boundaries.

Halo exchange:
- Each rank owns a local tile of the grid.
- Before update, it exchanges boundary rows/columns with adjacent ranks.
- Received boundary data is stored in halo/ghost cells.
- Then each cell counts infected neighbors using local + halo values.

Update idea:
- If a susceptible cell has k infected neighbors:
  p_infect = 1 - (1 - beta)^k
- Infected cells recover with probability gamma.

Interpretation:
- "Neighbor" means immediate spatial neighbors only.


3) Theoretical difference between the two codes

Mixing assumption:
- sir_gpu.py: homogeneous mixing (mean-field ODE-style compartment model).
- sir_neighbor_mpi.py: local-contact mixing on a graph/grid (stochastic cellular model).

State representation:
- sir_gpu.py: 3 aggregate compartments (S, I, R) per rank shard.
- sir_neighbor_mpi.py: state per cell (S/I/R) across a distributed grid.

Communication pattern:
- sir_gpu.py: global collectives (all ranks contribute each step).
- sir_neighbor_mpi.py: local neighbor communication (halo exchange with adjacent ranks), plus small global reductions for summary counts.

Epidemic dynamics:
- sir_gpu.py: smooth global curves; no wavefront/local clusters by construction.
- sir_neighbor_mpi.py: spatial spread, clustering, fronts, and locality effects.

Scalability characteristics:
- sir_gpu.py: low memory/state size, minimal per-step math, but global synchronization each step.
- sir_neighbor_mpi.py: higher memory/compute per rank, communication proportional to tile boundary, usually better locality.

When to use which:
- Use sir_gpu.py for fast what-if sweeps and coarse parameter studies.
- Use sir_neighbor_mpi.py when physical locality/contact structure matters.
