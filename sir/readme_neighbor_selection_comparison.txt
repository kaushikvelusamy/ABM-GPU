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





conv2d is efficient for this stencil because it turns neighbor counting into one dense, optimized GPU kernel instead of many per-cell branchy operations.

From your README (readme_neighbor_selection_comparison.txt):

Build infected mask X (1 infected, 0 otherwise).
Use a small kernel K (for 4-neighbor: cross shape, center 0).
Compute N = conv2d(X, K) to get infected-neighbor counts everywhere in parallel.
Apply infection rule from N (e.g., p=1-(1-beta)^k).
Why it’s fast on GPU:

conv2d uses highly tuned vendor libraries (cuDNN-like paths, cache/shared-memory optimized).
Work is regular and vectorized (good memory coalescing, fewer branches).
One kernel launch replaces multiple shift/sum ops.
How to distribute efficiently across GPUs:

Domain-decompose the 2D grid into tiles per GPU/rank.
Exchange 1-cell halos (N/S/E/W) each step before conv2d.
Run local conv2d on tile+halo; update interior and border consistently.
Use nonblocking halo exchange (Isend/Irecv) and overlap with interior compute.
Keep halo traffic as neighbor P2P; only use Allreduce for global S/I/R stats.
Minimize host staging; prefer CUDA-aware MPI/GPUDirect if available.
Key scaling rule: stencil comm cost grows with tile boundary (surface), compute grows with tile area (volume). So larger, well-shaped tiles and overlap are the main performance levers.





Simple 4-GPU picture

Use an 8x8 infected map X where:

1 = infected
0 = not infected
Example global grid (one timestep):

0 0 1 0 | 0 0 0 0
0 1 1 0 | 0 1 0 0
0 0 0 0 | 0 0 0 0
1 0 0 0 | 0 0 1 0
--------+--------
0 0 0 0 | 1 0 0 0
0 1 0 0 | 0 0 0 0
0 0 0 1 | 0 1 0 0
0 0 0 0 | 0 0 0 1
Split across 4 GPUs (2x2 tiles):

GPU0 = top-left 4x4
GPU1 = top-right 4x4
GPU2 = bottom-left 4x4
GPU3 = bottom-right 4x4
What each GPU does per step

Exchange 1-cell border (halo) with neighbor GPUs.
Run conv2d on local tile+halo using 4-neighbor kernel:
0 1 0
1 0 1
0 1 0
Output is neighbor-count N (how many infected neighbors each cell has).
For susceptible cells, compute infection probability from N (for example p = 1 - (1-beta)^N).
Tiny local example (GPU0 corner)

Suppose GPU0 has this local 4x4 infected mask (before halo):

0 0 1 0
0 1 1 0
0 0 0 0
1 0 0 0
For cell (row2,col2) (value 1 here, just for count demo), neighbors are up/down/left/right:

up 0, down 0, left 0, right 1 -> count 1
conv2d does this counting for all 16 cells at once in one optimized kernel call.

Why conv2d helps you

Without conv2d, you manually do multiple shifted reads/adds per cell (branchy, many kernels/ops).
With conv2d:

one library-optimized GPU op computes all neighbor counts
better memory access + kernel fusion in backend
easier code, usually faster and more stable performance
So the win is: less custom stencil code + faster neighbor counting, while MPI halos handle cross-GPU boundaries.


i didnt get the first part - What each GPU does per step
 0 1 0
1 0 1
0 1 0

That 3x3 matrix is the stencil kernel (weights used by conv2d):

0 1 0
1 0 1
0 1 0
It means for each cell, count only:

top (1)
left (1)
right (1)
bottom (1)
ignore diagonals (0)
ignore self (center 0)
So each GPU step is:

Get halos from neighbor GPUs (so edge cells can see top/left/right/bottom across GPU boundaries).
Run conv2d(X_local, kernel) on its tile.
Result is neighbor counts per cell.
Quick single-cell example:

Neighborhood around center C:

a b c
d C e
f g h
With that kernel, count = b + d + e + g (not a,c,f,h, not C).

If b=1, d=0, e=1, g=1, then count = 3.

That is exactly what the kernel encodes.



o each GPU step is:

Get halos from neighbor GPUs (so edge cells can see top/left/right/bottom across GPU boundaries).
Run conv2d(X_local, kernel) on its tile.
Result is neighbor counts per cell.

Perfect, let’s do one full timestep with your exact 8x8 example and 4 GPUs.

Global X (1=infected, 0=susceptible)

0 0 1 0 | 0 0 0 0
0 1 1 0 | 0 1 0 0
0 0 0 0 | 0 0 0 0
1 0 0 0 | 0 0 1 0
--------+--------
0 0 0 0 | 1 0 0 0
0 1 0 0 | 0 0 0 0
0 0 0 1 | 0 1 0 0
0 0 0 0 | 0 0 0 1
GPU split (2x2):

GPU0 = top-left 4x4
GPU1 = top-right 4x4
GPU2 = bottom-left 4x4
GPU3 = bottom-right 4x4
Kernel used by conv2d:

0 1 0
1 0 1
0 1 0
(only top/left/right/bottom count)

Halo exchange (start of timestep)
Example for GPU1:
Gets left halo from GPU0 (global col 4, rows 1..4): [0,0,0,0]
Gets bottom halo from GPU3 (global row 5, cols 5..8): [1,0,0,0]
Top/right are physical boundary => zeros.
Run conv2d on GPU1 tile
GPU1 local tile (rows 1..4, cols 5..8):
0 0 0 0
0 1 0 0
0 0 0 0
0 0 1 0
After conv2d with that kernel, neighbor-count map on GPU1 is:

0 1 0 0
1 0 1 0
0 1 1 0
1 1 0 1
Important boundary example:

Cell global (4,5) is on GPU1 bottom edge.
Count = up (3,5)=0 + down (5,5)=1 + left (4,4)=0 + right (4,6)=0 = 1.
That down value came from GPU3 halo. Without halo, you’d get wrong count.
Update state from counts
For each susceptible cell (X=0), use count k from conv output.
Example with beta=0.3: p_infect = 1-(1-0.3)^k.
For (4,5), k=1 => p_infect=0.3.
Apply random draw to decide if it flips 0 -> 1.
That is one complete timestep: exchange halos -> conv2d neighbor count -> infection/recovery update.


