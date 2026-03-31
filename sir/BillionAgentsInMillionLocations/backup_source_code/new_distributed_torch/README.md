# Moving-Agent SIR Model (PyTorch Distributed GPU Version)

This folder contains a self-contained distributed PyTorch version of the moving-agent SIR model:

- `sir_moving_agents_torch_distributed.py`
- `moving_sir_plotting.py`
- `plot_moving_sir_csv.py`

The model keeps the same moving-agent SIR idea as the single-process CPU version, but distributes the global grid across ranks with MPI communication through `mpi4py`. It is GPU-capable through PyTorch device placement, while still falling back to CPU if no accelerator is available.

## What The Code Does

`sir_moving_agents_torch_distributed.py`:

- creates a global `rows x cols` grid of locations
- partitions the grid by row blocks across distributed ranks
- initializes the full population on rank 0 and broadcasts it
- assigns each agent to the rank that owns its current row range
- stores each rank's local agents in PyTorch tensors on the selected device
- moves agents each step
- migrates agents across neighboring row partitions when movement crosses a rank boundary
- computes local occupancy and local S/I/R counts
- exchanges boundary infection information between neighboring ranks
- reduces global S/I/R summary values to rank 0
- writes:
  - rank 0 global `summary.csv`
  - one `rank<N>_agent_history.csv` per rank under `agent_history/`
  - one `rank<N>_location_history.csv` per rank under `location_history/`
  - rank 0 `run_config.json`
- optional rank 0 `summary.png`

The rank 0 run metadata and plot can also report:

- distributed runtime
- peak rank memory
- total GPU time
- peak GPU memory
- total output I/O time
- plot generation time
- total storage

## Default Configuration

The default run is:

- `rows = 100`
- `cols = 100`
- `num_agents = 10000`
- `infected0 = 10`
- `beta = 0.30`
- `gamma = 0.05`
- `steps = 30`
- `move_prob = 0.50`
- `infection_neighborhood = same_plus_news`
- `movement_neighborhood = stay_news`
- `seed = 0`

This means:

- `10000` global locations
- `10000` total agents
- `10` initially infected agents
- PyTorch distributed execution across however many ranks are launched

## Configuration Parameters

`sir_moving_agents_torch_distributed.py` accepts:

- `--rows`
  Global grid row count.
- `--cols`
  Global grid column count.
- `--num-agents`
  Total number of agents across all ranks.
- `--infected0`
  Number of initially infected agents.
- `--beta`
  Per-exposure infection intensity. For exposure count `k`, infection probability is `1 - (1 - beta)^k`.
- `--gamma`
  Per-step recovery probability for infected agents.
- `--steps`
  Number of simulated timesteps.
- `--move-prob`
  Probability that an agent attempts movement at a timestep.
- `--infection-neighborhood`
  Allowed values:
  `same_cell`, `same_plus_news`
- `--movement-neighborhood`
  Allowed values:
  `news`, `stay_news`
- `--seed`
  Random seed.
- `--out`
  Output prefix for flat files.
- `--run-dir`
  Output directory for a self-contained run folder.
- `--plot-summary`
  Generate a rank 0 PNG summary plot after the run.
- `--plot-out`
  Optional explicit PNG path.

## Moving-Agent Parameters

The moving-agent-specific controls are the same conceptual ones as the single-process version:

- `--rows`, `--cols`
  Define the global spatial grid.
  `num_locations = rows * cols`
- `--move-prob`
  Probability that an agent attempts movement on a step.
  Higher values mean faster spatial mixing.
- `--movement-neighborhood`
  `news` allows north, east, west, south movement.
  `stay_news` allows those moves plus staying in place.
- `--infection-neighborhood`
  `same_cell` means infection exposure is only from the exact same location.
  `same_plus_news` means the same location plus north, east, west, and south neighboring cells.
- `num_agents / (rows * cols)`
  Average density of agents per location.
  This is still one of the most important outbreak-shaping quantities.

## Compute Requirements And Scaling

This version is distributed with PyTorch, so scaling depends on:

- number of agents
- number of locations
- number of steps
- number of ranks
- amount of logging
- communication overhead at rank boundaries

At a high level:

- local computation scales with the number of agents and rows owned by each rank
- global summary reduction scales with the number of steps
- boundary communication scales with `cols` and the number of steps
- logging can dominate storage even if simulation memory is manageable

### Which Parts Run On CPU

- process launch and environment setup
- rank and world-size discovery from launcher environment variables
- MPI rank management and communication through `mpi4py`
- CSV writing
- JSON metadata writing
- plotting on rank 0
- MPI communication is currently staged through CPU-resident NumPy buffers for halo exchange, reductions, and agent migration

### Which Parts Run On GPU

If PyTorch selects `cuda`, `xpu`, or `mps`, the following local rank state lives on that device:

- local agent id tensor
- local agent row and column tensors
- local state tensor
- local occupancy and local S/I/R count grids
- local exposure and infection-probability tensors
- local transition masks such as `became_infected` and `became_recovered`
- local movement and infection updates

So the per-rank local simulation state is device-resident, while inter-rank communication is handled through MPI buffers on CPU.

### How Communication Is Handled

This is the most important distributed detail.

The current implementation uses:

- `mpi4py`
- row-wise domain decomposition
- nearest-neighbor communication only in the vertical direction
- MPI reductions for global summary values
- MPI `Sendrecv` for row-boundary infection data
- MPI `allgather` of migration payloads for agents crossing neighboring rank boundaries

More specifically:

- each rank owns a contiguous block of global rows
- if an agent moves above or below its local row block, it migrates to the neighboring rank
- when `infection_neighborhood=same_plus_news`, a rank needs infected counts from the top and bottom neighboring ranks to compute exposure at its boundary rows
- horizontal neighbors are local within the same rank because the decomposition is by rows, not by columns
- global S/I/R values, occupied-location counts, and moved-agent counts are reduced to rank 0 each step

Important implication:

- communication volume grows with `cols` and with how many agents cross row boundaries
- a taller grid split across many ranks increases communication frequency at partition edges
- the code is simpler than a full 2D decomposition, but it is not communication-free

### How The Problem Is Divided Across GPUs

The job is split by grid rows, with one MPI rank per GPU. Since you launched `24` ranks, you got `24` `agent_history.csv` files and `24` `location_history.csv` files, one pair per rank/GPU.

In the code, the split happens through `split_rows(...)`. Each rank owns a contiguous slab of the global grid:

- rank `0` gets the first block of rows
- rank `1` gets the next block
- and so on

For a `100 x 100` grid on `24` ranks:

- `100 // 24 = 4`
- `100 % 24 = 4`

So the row ownership is:

- ranks `0-3`: `5` rows each
- ranks `4-23`: `4` rows each

That means:

- the first `4` GPUs each own `5 x 100 = 500` locations
- the remaining `20` GPUs each own `4 x 100 = 400` locations

Agents are also assigned by row ownership. After rank `0` creates the initial global population, each rank keeps only the agents whose current global row falls inside its row block.

During the run:

- if an agent moves within the same row slab, it stays on the same rank/GPU
- if it crosses the top or bottom boundary of that slab, it migrates to the neighboring rank through MPI

So the `24` files in each subfolder mean:

- `agent_history/rankK_agent_history.csv`: all agents currently owned by rank `K` over time
- `location_history/rankK_location_history.csv`: all occupied cells in rank `K`'s row slab over time

The run is not split by agent ID range and not replicated on every GPU. It is a spatial decomposition of the grid.

For this `100 x 100` on `24` ranks example, the row layout is:

```text
Rank 0: rows [0, 5)
Rank 1: rows [5, 10)
Rank 2: rows [10, 15)
Rank 3: rows [15, 20)
Rank 4: rows [20, 24)
...
Rank 23: rows [96, 100)
```

### Costliest Computation

The costliest recurring computation is usually the per-step agent update work:

- agent movement decisions
- exposure lookup for every local agent
- infection probability computation for susceptible local agents
- recovery update for infected local agents

In this implementation, that local per-agent computation is performed on the selected PyTorch device, so on a GPU-enabled run it is primarily GPU work.

The most expensive communication-related cost is usually:

- boundary halo exchange of infected counts
- plus agent migration when many agents cross row partitions

That communication path is currently MPI-based and staged through CPU/NumPy buffers.

### Main In-Memory State

Per rank, the main local state includes:

- local agent ids
- local agent rows
- local agent cols
- local states
- movement bookkeeping arrays
- local occupancy grid
- local susceptible, infected, and recovered count grids
- local exposure and infection-probability arrays

Global arrays are not permanently stored on every rank.
Only rank 0 briefly constructs the initial global population before broadcasting it.

### Data Structures And Per-Member Size

The main PyTorch tensors use approximately these payload sizes per element:

- `agent_ids`
  dtype: `int64`
  size per element: `8 bytes`
- `agent_rows`
  dtype: `int64`
  size per element: `8 bytes`
- `agent_cols`
  dtype: `int64`
  size per element: `8 bytes`
- `states`
  dtype: `int8`
  size per element: `1 byte`
- `moved`
  dtype: `bool`
  size per element: `1 byte`
- `exposure`
  dtype: `int64`
  size per element: `8 bytes`
- `infection_prob`
  dtype: `float64`
  size per element: `8 bytes`
- `became_infected`
  dtype: `bool`
  size per element: `1 byte`
- `became_recovered`
  dtype: `bool`
  size per element: `1 byte`
- `occupancy`, `s_counts`, `i_counts`, `r_counts`
  dtype: `int64`
  size per element: `8 bytes`
  shape per rank: `local_rows x cols`

Useful shorthand:

- one `int64` local agent array costs about `8 * local_num_agents` bytes
- one `bool` local agent array costs about `1 * local_num_agents` bytes
- one `float64` local agent array costs about `8 * local_num_agents` bytes
- one `int64` local grid costs about `8 * local_rows * cols` bytes

Because the run is distributed:

- agent arrays scale with local agents on that rank
- grid arrays scale with the local row partition on that rank
- the largest rank usually determines peak memory

### Logging Cost

Output storage usually grows faster than core simulation memory.

- `summary.csv`
  Scales roughly with `O(steps)`
- `agent_history/rank<N>_agent_history.csv`
  For rank `N`, scales roughly with `O(steps * local_num_agents_on_rank)`
- total agent-history storage across all ranks
  Scales roughly with `O(steps * num_agents)`
- `location_history/rank<N>_location_history.csv`
  For rank `N`, scales roughly with `O(steps * local_occupied_locations_on_rank)`
- total location-history storage across all ranks
  Worst-case rough upper bound:
  `O(steps * rows * cols)`
- `run_config.json`
  Small
- `summary.png`
  Small compared with history CSV files

### Quick Estimation

A simple first-order estimate is:

- local compute per rank
  `~ steps * (local_num_agents + local_rows * cols)`
- total agent-history rows across all ranks
  `~ (steps + 1) * num_agents`
- total location-history rows across all ranks
  `<= (steps + 1) * rows * cols`
- boundary communication per step
  roughly proportional to `cols`

If the grid is split evenly across `world_size` ranks:

- `local_rows ~ rows / world_size`
- average local agents per rank
  `~ num_agents / world_size`

So compute and memory per rank usually improve as ranks increase, but communication overhead also increases.

## Why Host Memory Can Stay High While GPU Memory Stays Low

You may see output like this:

```text
Total wall time: 32.080 s
Host memory per rank (peak): 1706.949 MB
Total host memory (all ranks): 40869.953 MB
GPU compute time on slowest rank: 30.944 s (96.5% of wall time)
GPU compute time (all ranks summed): 681.306 s
GPU memory per rank (peak): 0.095 MB
Total GPU memory (all ranks): 1.875 MB
```

A few things are going on here.

The short version is:

- the code is much better than the old full-population broadcast path
- but a run like `100 x 100` with `10k` agents on `24` ranks is still very small on-GPU
- `Host memory per rank` is measuring the entire Python + MPI + PyTorch/XPU process RSS, not just the model arrays

So the high CPU memory is likely dominated by runtime overhead and host-side staging, not by the agent state itself.

The main reasons are:

1. `ru_maxrss` is process memory, not model-only memory.

- `Host memory per rank (peak)` includes:
  - Python interpreter
  - imported modules
  - PyTorch runtime
  - XPU runtime / oneAPI runtime
  - MPI runtime
  - file I/O buffers
  - temporary NumPy arrays
  - allocator caches

So `1706 MB` does not mean the local agents alone are using `1.7 GB`.

2. This problem can still be tiny on the GPU.

For `100 x 100` with `10k` agents on `24` ranks, each rank only owns about:

- `4-5` rows
- `400-500` cells
- about `10000 / 24 ~= 417` agents

That is far too small to drive noticeable GPU memory use. The actual model tensors on device are tiny.

3. The code still does a lot of CPU staging every step.

Even after fixing initialization, each step still does host-side work for:

- MPI halo exchange via NumPy buffers
- agent migration packing and unpacking through NumPy and Python
- CSV logging through CPU conversions
- repeated `.to("cpu").numpy()` style extraction for writing logs and MPI

Those temporary host arrays can increase RSS and allocator footprint.

4. XPU memory accounting may not reflect all accelerator-related allocations.

`GPU memory per rank` comes from PyTorch's tracked XPU allocator. That usually means:

- it counts PyTorch-managed device allocations
- it may not include every runtime-side buffer
- some Intel/XPU paths may use shared or host-visible memory that shows up more in process RSS than in the reported device peak

So some accelerator-adjacent memory may still appear as host memory.

5. Framework baseline can be large on Aurora-style stacks.

With `module load frameworks`, a single Python + PyTorch + XPU + MPI process can already have a substantial baseline footprint before the model data becomes important. Across `24` ranks, that baseline multiplies quickly.

Why time improved but memory did not drop dramatically:

- removing the full broadcast helped startup cost and scalability
- but a small run means fixed per-rank runtime overhead can dominate the memory picture
- the GPU may be busy in compute time, but not storing much data

What is likely dominating host memory now:

- runtime and framework overhead per rank
- CPU staging for MPI
- CPU staging for CSV logging
- temporary NumPy conversions
- allocator caching

What would reduce host memory further:

- reduce or disable detailed per-step CSV logging
- avoid converting tensors to CPU every step unless needed
- replace Python and NumPy migration packing with more compact tensor-based exchange
- reduce temporary host buffers for MPI
- profile memory around:
  - startup
  - first logging write
  - first migration
  - first halo exchange

What would increase GPU memory use in a meaningful way:

- much larger local problem per rank
- fewer ranks for the same global problem
- larger local agent count
- larger local grids
- more device-resident state and less CPU staging

So the main answer is:

- GPU memory is low because the per-rank model state can be genuinely tiny
- CPU memory is high because RSS includes the whole software stack plus host staging and logging overhead

If you want to reduce host RSS further, a useful next step is to add lighter logging modes such as:

- `--log-agent-history 0`
- `--log-location-history 0`
- `--log-every N`

That would likely make the CPU-memory picture much cleaner for large runs.

## How To Run

### Single-Rank Smoke Test

```bash
cd /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/new_distributed_torch
python3 sir_moving_agents_torch_distributed.py --run-dir results/smoke_test --plot-summary
```

### `torchrun` Launch

```bash
cd /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/new_distributed_torch
torchrun --standalone --nproc_per_node=4 \
  sir_moving_agents_torch_distributed.py \
  --rows 100 \
  --cols 100 \
  --num-agents 10000 \
  --run-dir results/torchrun_4ranks \
  --plot-summary
```

### `mpiexec` Launch Pattern

If you want to launch with `mpiexec` in your environment, the code now detects rank information from MPI-style environment variables such as `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, and `MPI_LOCALRANKID`.

Your launch style can be documented like this:

```bash
module load frameworks

NNODES=$(wc -l < "$PBS_NODEFILE")
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))
CPU_BINDING1=list:4:9:14:19:20:25:56:61:66:71:74:79

echo "NUM_OF_NODES=${NNODES} TOTAL_NUM_RANKS=${NRANKS} RANKS_PER_NODE=${RANKS_PER_NODE} CPU_BINDING1=${CPU_BINDING1}"

cd /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/new_distributed_torch

mpiexec -np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind ${CPU_BINDING1} -genvall \
  python3 sir_moving_agents_torch_distributed.py \
    --rows 100 \
    --cols 100 \
    --num-agents 10000 \
    --infected0 10 \
    --beta 0.30 \
    --gamma 0.05 \
    --steps 30 \
    --move-prob 0.50 \
    --infection-neighborhood same_plus_news \
    --movement-neighborhood stay_news \
    --run-dir results/mpi_run_100x100_10kagents \
    --plot-summary
```

Notes for this launch mode:

- rank, world size, and local rank are detected from MPI-style launcher environment variables
- communication is handled with `mpi4py`
- if you later want direct accelerator-aware collectives, the communication layer could be migrated to oneCCL, but this implementation is currently MPI-based

## Rules

- the model is row-partitioned, not 2D-tiled
- horizontal neighbor effects are local to a rank
- vertical neighbor effects require communication
- rank 0 writes the global summary and run config
- every rank writes its own agent-history and location-history CSV files
- plotting happens only on rank 0
- if PyTorch is unavailable, the script will not run
- if `matplotlib` is unavailable, the simulation can still complete but the plot is skipped
