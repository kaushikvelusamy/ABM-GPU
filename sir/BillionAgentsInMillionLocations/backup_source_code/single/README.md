# Moving-Agent SIR Model (CPU Only)

This backup folder documents the non-distributed moving-agent SIR model in:

- `single/sir_moving_agents.py`
- `single/plot_moving_sir_csv.py`
- `single/moving_sir_plotting.py`

The code is a single-process, CPU-only agent-based SIR simulation on a 2D grid. Agents move between grid cells, infected agents expose nearby susceptible agents, and infected agents recover with probability `gamma` each step. The run writes CSV logs plus a JSON config file, and it can also generate a summary PNG with S/I/R curves in the same overall style as the reference figure `20260320-211501-003.jpg`.

## Requirements

- Python 3
- `numpy`
- `matplotlib` only if you want PNG plots

## What The Code Does

`single/sir_moving_agents.py`:

- creates a `rows x cols` location grid
- randomly places `num_agents` agents on the grid
- marks `infected0` agents as initially infected
- updates the model for `steps` discrete timesteps
- moves agents with probability `move_prob`
- computes infection exposure from either the same cell or the same cell plus the four cardinal neighbors
- recovers infected agents with probability `gamma`
- writes detailed outputs for later analysis and plotting
- records total runtime and peak process memory for the run

## How This Differs From A Traditional Non-Moving SIR Model

In a traditional non-moving SIR model, people are usually assumed to mix in a fixed way:

- fully mixed compartment model
  Every susceptible person can effectively interact with the whole infected population through aggregate rates
- static network or static groups
  People interact with a fixed set of contacts, and those contacts do not move in space during the run
- no explicit location state
  The model tracks S, I, and R counts, but not where agents are

In this moving-agent model, the simulation explicitly tracks where every agent is on a 2D grid and lets exposure change over time because agents move:

- each agent has a location `(row, col)` at every step
- infection risk depends on who is nearby at that step, not just on global counts
- contact patterns are dynamic because movement changes local neighborhoods over time
- density matters
  The same number of agents can behave very differently depending on how many locations they are spread across
- mobility matters
  Increasing movement can accelerate mixing and make outbreaks peak earlier

Conceptually, this means the moving-agent model adds a spatial layer on top of SIR:

- classic non-moving model
  "How many susceptible and infected people exist?"
- moving-agent model
  "How many susceptible and infected people exist, where are they, and who is near whom right now?"

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

- `10000` grid locations
- `10000` total agents
- `10` initially infected agents
- CPU-only execution in one Python process

## Configuration Parameters

`single/sir_moving_agents.py` accepts:

- `--rows`
  Grid row count.
- `--cols`
  Grid column count.
- `--num-agents`
  Total number of agents in the simulation.
- `--infected0`
  Number of initially infected agents.
- `--beta`
  Per-exposure infection intensity. For a susceptible agent with exposure count `k`, infection probability is `1 - (1 - beta)^k`.
- `--gamma`
  Per-step recovery probability for infected agents.
- `--steps`
  Number of simulated timesteps.
- `--move-prob`
  Probability that an agent attempts a move at a timestep.
- `--infection-neighborhood`
  Exposure rule.
  Allowed values:
  `same_cell`, `same_plus_news`
- `--movement-neighborhood`
  Movement rule.
  Allowed values:
  `news`, `stay_news`
- `--seed`
  Random seed for reproducible runs.
- `--out`
  Output prefix for flat files such as `name_summary.csv`.
- `--run-dir`
  Output directory for a self-contained run folder.
- `--plot-summary`
  Generate a PNG summary plot after the simulation finishes.
- `--plot-out`
  Optional PNG path for the generated plot.

## Moving-Agent Parameters

The parameters below are the ones that make this model different from a non-moving or fully mixed SIR model.

- `--rows`, `--cols`
  These define the 2D grid of locations. Together they set the number of possible places agents can occupy:
  `num_locations = rows * cols`
  Increasing the grid while keeping `num_agents` fixed lowers average density.

- `--move-prob`
  Probability that an agent attempts movement on a step.
  Higher values mean faster mixing.
  Lower values mean agents stay near their current local neighborhoods longer.

- `--movement-neighborhood`
  Controls where an agent may move when it attempts motion.
  `news` means north, east, west, or south.
  `stay_news` means the same four directions plus staying in place.
  This parameter changes how aggressively mobility spreads agents across space.

- `--infection-neighborhood`
  Controls which nearby infected counts contribute to exposure.
  `same_cell` means only agents in the exact same location matter.
  `same_plus_news` means the same location plus the four cardinal neighboring cells also contribute.
  This makes infection more local or less local depending on the choice.

- `--num-agents / (rows * cols)`
  This is not a direct input flag, but it is one of the most important moving-agent quantities.
  It is the average number of agents per location.
  A larger value means denser local mixing and usually faster spread.

## Compute Requirements And Scaling

This code is CPU-only and single-process, so scaling behavior is dominated by:

- number of agents
- number of locations
- number of steps
- how much output logging you keep

### Main In-Memory State

The core simulation stores arrays such as:

- agent rows
- agent cols
- agent states
- per-step exposure and infection arrays
- location occupancy and S/I/R count grids

The most important scaling terms are:

- agent state scales roughly with `O(num_agents)`
- grid state scales roughly with `O(rows * cols)`
- total simulation work scales roughly with `O(steps * (num_agents + rows * cols))`

In practice:

- if you double `num_agents`, runtime usually grows by about 2x or a bit more
- if you double both `rows` and `cols`, grid-related work and grid memory grow by about 4x
- if you double `steps`, runtime and output size both grow by about 2x

### Data Structures And Per-Member Size

The current implementation uses NumPy arrays with the following dtypes.
These byte counts are the payload size per element, not counting small Python or NumPy container overheads.

- `agent_rows`
  dtype: `int64`
  size per element: `8 bytes`
  total payload: `8 * num_agents`

- `agent_cols`
  dtype: `int64`
  size per element: `8 bytes`
  total payload: `8 * num_agents`

- `states`
  dtype: `int8`
  size per element: `1 byte`
  total payload: `1 * num_agents`

- `old_rows`
  dtype: `int64`
  size per element: `8 bytes`
  total payload: `8 * num_agents`
  This is a temporary copy used during movement logging.

- `old_cols`
  dtype: `int64`
  size per element: `8 bytes`
  total payload: `8 * num_agents`
  This is a temporary copy used during movement logging.

- `moved`
  dtype: `bool`
  size per element: `1 byte`
  total payload: `1 * num_agents`

- `exposure`
  dtype: effectively `int64`
  size per element: `8 bytes`
  total payload: `8 * num_agents`

- `infection_prob`
  dtype: `float64`
  size per element: `8 bytes`
  total payload: `8 * num_agents`

- `became_infected`
  dtype: `bool`
  size per element: `1 byte`
  total payload: `1 * num_agents`

- `became_recovered`
  dtype: `bool`
  size per element: `1 byte`
  total payload: `1 * num_agents`

- `occupancy`
  dtype: `int64`
  shape: `rows x cols`
  size per element: `8 bytes`
  total payload: `8 * rows * cols`

- `s_counts`
  dtype: `int64`
  shape: `rows x cols`
  size per element: `8 bytes`
  total payload: `8 * rows * cols`

- `i_counts`
  dtype: `int64`
  shape: `rows x cols`
  size per element: `8 bytes`
  total payload: `8 * rows * cols`

- `r_counts`
  dtype: `int64`
  shape: `rows x cols`
  size per element: `8 bytes`
  total payload: `8 * rows * cols`

- `exposure_grid`
  dtype: `int64`
  shape: `rows x cols`
  size per element: `8 bytes`
  total payload: `8 * rows * cols`
  This is derived from infected counts each step.

- `cardinal_neighbor_sum(i_counts)` output
  dtype: `int64`
  shape: `rows x cols`
  size per element: `8 bytes`
  total payload: `8 * rows * cols`
  This is another temporary grid used when `infection_neighborhood=same_plus_news`.

Useful shorthand:

- one `int64` agent array costs about `8 * num_agents` bytes
- one `float64` agent array costs about `8 * num_agents` bytes
- one boolean agent array costs about `1 * num_agents` bytes
- one `int64` grid costs about `8 * rows * cols` bytes

Example at the default scale:

- `num_agents = 10000`
  one `int64` agent array is about `80000 bytes`, about `78.1 KB`
- `rows = cols = 100`
  one `int64` grid is also about `80000 bytes`, about `78.1 KB`

So even before temporary arrays and CSV buffering, a default run keeps several arrays of roughly that size alive at once.
At much larger scales, the repeated agent-history logging usually becomes the bigger practical limit, but these array sizes are the right first estimate for core simulation memory.

### Logging Cost

The output files can become much larger than the in-memory model state.

- `summary.csv`
  Scales roughly with `O(steps)`
- `agent_history.csv`
  Scales roughly with `O(steps * num_agents)`
  This is usually the dominant file
- `location_history.csv`
  Scales roughly with `O(steps * occupied_locations)`
  In the worst case this approaches `O(steps * rows * cols)`
- `run_config.json`
  Essentially constant size
- `summary.png`
  Small compared with CSV logs

### Quick Estimation Rules

For rough planning:

- compute work estimate
  `work ~ steps * (num_agents + rows * cols)`
- densification estimate
  `average_agents_per_location = num_agents / (rows * cols)`
- largest log estimate
  `agent_history rows = (steps + 1) * num_agents`
- upper bound for location log
  `location_history rows <= (steps + 1) * rows * cols`

Examples:

- `100 x 100`, `10000` agents, `30` steps
  `num_locations = 10000`
  `average_agents_per_location = 1`
  `agent_history rows = 31 * 10000 = 310000`
- `1000 x 1000`, `1000000` agents, `30` steps
  `num_locations = 1000000`
  `average_agents_per_location = 1`
  `agent_history rows = 31 * 1000000 = 31000000`
  This will make the per-agent CSV extremely large even before plotting

### How To Estimate Before A Big Run

The safest workflow is:

1. run a smaller case with the same ratios
2. note `CPU runtime`, `CPU peak memory`, and `Total storage`
3. scale those measurements using the formulas above

For example, if you keep:

- the same `steps`
- the same `num_agents / (rows * cols)` density
- the same logging options

then a first-order estimate is:

- runtime grows about linearly with `num_agents`
- memory grows about linearly with `num_agents + rows * cols`
- `agent_history.csv` grows linearly with `steps * num_agents`
- `location_history.csv` grows roughly with how many cells become occupied over time

### Practical Advice

- for exploratory runs, reduce `num_agents` first
- if you only need epidemic curves, `summary.csv` is cheap but `agent_history.csv` is expensive
- large grids with the same number of agents reduce density and may also reduce occupied-location logging
- very large `steps` values increase both compute time and output size even if the epidemic has already stabilized
- use a small pilot run to calibrate scaling instead of jumping directly to a very large case

### Sizing Cookbook

These are practical starting points for this single-process CPU code.

- small smoke test
  `rows=20`, `cols=20`, `num_agents=200`, `steps=10-30`
  Good for checking correctness, logging, and plotting.
  Usually finishes quickly and keeps CSV sizes small.

- small development run
  `rows=100`, `cols=100`, `num_agents=10000`, `steps=30`
  This is a good baseline for tuning parameters and comparing epidemic shapes.
  Output sizes are still manageable on a laptop or workstation.

- medium scaling run
  `rows=300`, `cols=300`, `num_agents=100000`, `steps=30`
  Useful for testing denser logs and more realistic agent counts.
  Expect noticeably larger `agent_history.csv` and longer runtime.

- large single-node stress test
  `rows=1000`, `cols=1000`, `num_agents=1000000`, `steps=30`
  This is mainly for strong CPU and memory environments.
  The simulation may still fit, but CSV logging can become very large and may dominate storage.

Rules of thumb:

- if `agent_history.csv` becomes too large, reduce `num_agents` or `steps` first
- if spread looks too fast, increase `rows * cols` or reduce `move_prob`
- if runtime is acceptable but storage is not, the per-agent log is usually the first bottleneck
- before moving to a larger case, try increasing scale by about `5x` to `10x` rather than jumping directly by `100x`

## Output Files

Using `--run-dir some_folder` creates:

- `summary.csv`
  Global S/I/R counts and occupancy statistics by timestep.
- `agent_history.csv`
  One row per agent per timestep with state, movement, exposure, and transition flags.
- `location_history.csv`
  One row per occupied location per timestep with occupancy and local S/I/R counts.
- `run_config.json`
  The full configuration used for the run, including runtime and peak memory.
- `summary.png`
  Optional plot written when `--plot-summary` is enabled. The annotation box includes runtime and peak memory when available.

At the end of a run, the console also prints each saved file with a human-readable size such as `12.45 KB` or `3.18 MB`.

Using `--out prefix_name` creates:

- `prefix_name_summary.csv`
- `prefix_name_agent_history.csv`
- `prefix_name_location_history.csv`
- `prefix_name_run_config.json`
- `prefix_name_summary.png` if plotting is enabled

## How To Run

Run from the `single` directory:

```bash
cd /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/single
python3 sir_moving_agents.py --run-dir ../outputs/single/default_cpu_run --plot-summary
```

That produces a self-contained CPU-only run folder in `../outputs/single/default_cpu_run/`.

If you prefer flat output filenames:

```bash
cd /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/single
python3 sir_moving_agents.py --out ../outputs/single/moving_sir_cpu --plot-summary
```

Example custom run:

```bash
cd /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/single
python3 sir_moving_agents.py \
  --rows 100 \
  --cols 100 \
  --num-agents 10000 \
  --infected0 25 \
  --beta 0.30 \
  --gamma 0.05 \
  --steps 30 \
  --move-prob 0.50 \
  --infection-neighborhood same_plus_news \
  --movement-neighborhood stay_news \
  --seed 7 \
  --run-dir ../outputs/single/exp1_cpu_only \
  --plot-summary
```

## Replot An Existing Summary CSV

If a run already produced `summary.csv`, you can regenerate the PNG:

```bash
cd /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/single
python3 plot_moving_sir_csv.py \
  --csv ../outputs/single/exp1_cpu_only/summary.csv \
  --config ../outputs/single/exp1_cpu_only/run_config.json \
  --out ../outputs/single/exp1_cpu_only/summary.png
```

If `run_config.json` is provided or auto-detected, the plot annotation includes:

- runtime in seconds
- peak process memory in MB

## Notes

- `step` is a discrete simulation timestep. If you want an epidemiology interpretation, it is reasonable to read one step as one day.
- `same_plus_news` means the same location plus north, east, west, and south neighboring cells.
- `stay_news` means an agent may stay in place or move one cell in a cardinal direction when it attempts movement.
- This README intentionally documents only the non-distributed CPU path in `backup_source_code/single`.
