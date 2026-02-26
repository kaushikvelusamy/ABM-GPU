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


4) Halo types used in distributed stencil/ABM codes

Common halo patterns:
- 1-cell face halo (most common): exchange only top/bottom rows and left/right columns.
- k-cell halo (radius r): exchange r layers when the stencil reaches farther than immediate neighbors.
- Corner-inclusive halo: include diagonal corner cells when using 8-neighbor (Moore) or larger diagonal stencils.
- Periodic halo: domain wraps around (torus), so physical edges exchange with opposite-side ranks.
- Non-periodic halo: physical boundaries are filled by boundary condition rules (fixed, reflective, absorbing, etc.).

Cost intuition:
- Communication grows with boundary size (surface area), not tile volume.
- Larger stencil radius increases halo width and message size.


5) Stencils and neighborhood definitions

Typical 2D stencils:
- Von Neumann (4-neighbor): N, S, E, W. (used by sir_neighbor_mpi.py)
- Moore (8-neighbor): 4-neighbor + diagonals.
- Cross/ring radius-r: all cells within Manhattan distance <= r.
- Square radius-r: all cells in a (2r+1)x(2r+1) window (often with center excluded).
- Weighted stencils: neighbors contribute with distance/direction-specific weights.

ABM interpretation:
- Stencil choice encodes contact structure.
- Tight stencil -> slower/local spread and stronger clustering.
- Wider stencil -> faster mixing and behavior closer to mean-field.


6) How conv2d is used for neighbor counting

The stencil update can be written as a convolution:
- Build binary infected map X (1 if infected else 0).
- Choose kernel K matching neighborhood shape.
- Compute neighbor count N = conv2d(X, K) with center weight set to 0.

Examples:
- 4-neighbor kernel:
  [0 1 0
   1 0 1
   0 1 0]
- 8-neighbor kernel:
  [1 1 1
   1 0 1
   1 1 1]

Why this is useful:
- GPU libraries optimize conv kernels heavily.
- One conv call can replace many manual shift/add operations.
- Boundary behavior is controlled by padding mode (zero, circular, reflect, explicit halo).

MPI + conv2d:
- Each rank still needs halo exchange before local conv2d near tile boundaries.
- Local interior can be updated independently once halos are current.


7) Neighbor selection options in HPC ABM practice

Grid/lattice neighborhoods:
- Fixed geometric neighborhoods (4/8/radius-r stencils).
- Directional/anisotropic neighborhoods (e.g., stronger east-west coupling).

Graph-based neighborhoods:
- Static graph (social/contact network).
- Dynamic graph (mobility/contact changes over time).

Hybrid neighborhoods:
- Local spatial contacts + random long-range contacts ("small-world" mixing).
- Multi-scale neighborhoods (household + workplace + community layers).

Probabilistic/weighted contact rules:
- Distance-decay contact probability.
- Type-based weighting (age group, location type, mobility class).

Parallelization implications:
- Regular grids map well to domain decomposition + halos.
- Irregular graphs often use partitioning + ghost vertices + sparse communication.
- Dynamic neighborhoods increase communication complexity and load imbalance risk.


8) P2P vs collectives (MPI and CCL/XCCL) for halo-based SIR

Core rule:
- Halo exchange is naturally neighbor point-to-point (P2P), not a global collective.
- Global monitoring/statistics (total S/I/R) are naturally collectives (allreduce).

MPI for this workload:
- P2P halo phase:
  - Use `Sendrecv` / `Isend`+`Irecv` (or neighborhood collectives like `Neighbor_alltoallv`).
  - Communicate only with N/S/E/W rank neighbors.
  - This is exactly what `sir_neighbor_mpi.py` does for boundary rows/columns.
- Collective phase:
  - Use `Allreduce` for global S/I/R summaries each step (small scalars).
  - Optionally reduce less frequently if only periodic logging is needed.

Why P2P is preferred for halos:
- Communication volume is local and sparse (4 neighbors in 2D Cartesian split).
- Global collectives for halo data would over-communicate and add unnecessary synchronization.

Where CCL/XCCL (NCCL/oneCCL, etc.) fits:
- Best suited for dense GPU collectives (allreduce/allgather/reduce-scatter), common in DL training.
- Less natural for fine-grained stencil halos unless the library and runtime provide efficient neighbor exchanges for your topology.

GPU-specific note for this code:
- Current GPU path stages halo buffers through host memory for MPI:
  GPU -> CPU -> MPI P2P -> CPU -> GPU.
- This keeps implementation simple/portable in Python but adds copy overhead.
- If available, CUDA-aware MPI/GPUDirect can reduce staging cost by sending device buffers directly.

Practical guidance:
- For stencil ABM (this file): keep halos as MPI neighbor P2P; keep global counts as `Allreduce`.
- Consider CCL/XCCL mainly if your bottleneck is large GPU collectives, not small neighbor halos.
- For very large GPU stencil runs, first evaluate CUDA-aware MPI + nonblocking halo overlap before redesigning around CCL.
