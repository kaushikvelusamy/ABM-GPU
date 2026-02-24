Schelling ABM on CPU / NVIDIA GPU / Intel Data Center GPU Max (PyTorch)
=====================================================================


python3 abm_schelling_gpu.py \
  --mode animate \
  --L 300 \
  --density 0.95 \
  --frac-A 0.5 \
  --threshold 0.70 \
  --minutes 2 \
  --fps 20 \
  --interval-ms 50 \
  --seed 7 \
  --gif schelling_big_clusters.gif

python3 abm_schelling_gpu.py \
  --mode animate \
  --minutes 3 \
  --fps 20 \
  --interval-ms 50 \
  --L 256 \
  --density 0.92 \
  --frac-A 0.5 \
  --threshold 0.6 \
  --gif schelling_3min.gif



What this code does
------------------
This script runs a simple Schelling-style agent-based model (ABM) on a 2D lattice.

- The world is an L×L grid.
- Each cell is one of:
  - 0 = EMPTY
  - 1 = Group A agent
  - 2 = Group B agent
- Agents evaluate their local neighborhood (8 surrounding cells, Moore neighborhood).
- If an agent is "unhappy" (too few same-type neighbors), it moves to a random empty cell.
- Repeat for many steps and observe emergent clustering/segregation.

The model being solved / studied
--------------------------------
Schelling’s segregation model is a canonical example of “micro-to-macro” emergence:

- Micro rule: each agent has a simple local preference (“I want at least T fraction of my neighbors
  to be like me”).
- Macro outcome: even mild preferences can yield strong clustering/segregation over time.

This code is useful when you want to:
- run large grids (big L) or many steps quickly,
- do many repeated runs (ensembles / parameter sweeps),
- prototype variants (different thresholds, neighborhoods, movement rules).

How the code is structured (high-level)
---------------------------------------
1) pick_device()
   Chooses where tensors live and where computation runs:
   - "cuda" if an NVIDIA GPU is available
   - "xpu" if an Intel GPU is available (e.g., Intel Data Center GPU Max)
   - otherwise "cpu"

2) init_grid()
   Initializes a random grid with a target occupancy (density) and group mix (frac_A).
   This corresponds to a random initial condition in ABM terms.

3) neighbor_counts()
   Computes, for every cell, how many neighbors are A and how many neighbors are B.
   This is the expensive part of many lattice ABMs, because it touches every cell each step.

4) step_schelling()
   For one timestep:
   - compute neighborhood counts (A_nb, B_nb)
   - compute each occupied cell’s similarity fraction
   - mark agents as unhappy if similarity < threshold
   - relocate unhappy agents into empty cells

5) run_schelling()
   Runs multiple steps and prints summary statistics periodically.

6) plot_schelling() / animate_schelling()
   Uses matplotlib to visualize the final state or animate updates.

Key ABM definitions in this implementation
------------------------------------------
Neighborhood
  Moore neighborhood (8-neighbor): the 3×3 neighborhood around a cell excluding the center.

Similarity / satisfaction
  For an agent, “similar neighbors” means neighbors of the same type (A sees A, B sees B).
  similarity = (# similar neighbors) / (# occupied neighbors)
  threshold = minimum similarity required to be “happy”

Movement rule (important)
  This implementation moves unhappy agents to random empty locations in the full grid.
  (This is the “global relocation” style movement.)

Important note for ABM experts: “movement” is the irregular part. The neighbor counting is regular.
GPUs accelerate regular work very well; irregular scatter/gather is typically harder to optimize.

What “convolution” is doing here (the neighbor count trick)
-----------------------------------------------------------
The expensive per-step operation is: “for each cell, sum up a function of its neighbors.”

Instead of looping in Python over every cell, the code expresses neighbor counting as a 3×3 stencil
computed by a 2D convolution:

Kernel (k):
    1 1 1
    1 0 1
    1 1 1

- Multiplying by this kernel and summing over a 3×3 window gives the number of True cells
  in the 8 neighbors around each cell (center excluded by the 0).

Implementation details:
- A_mask = (grid == A)  converted to float
- B_mask = (grid == B)  converted to float
- F.pad(..., mode="circular") pads the grid so that boundaries wrap around (toroidal world).
- F.conv2d computes neighbor sums for every cell in a single optimized kernel call.

Why PyTorch is used (even though this is not ML)
------------------------------------------------
PyTorch is used as a high-performance tensor runtime:

- A tensor can live on CPU, NVIDIA GPU ("cuda"), or Intel GPU ("xpu").
- The same Python code runs on different devices by changing where the tensors live.
- It provides a highly optimized conv2d implementation, which is exactly what we want for neighbor sums.

So PyTorch here is essentially:
- “NumPy-like arrays + fast kernels + GPU support + a device abstraction”
not “deep learning training.”

What runs on the GPU vs CPU
---------------------------
If grid/state tensors are created on GPU (cuda or xpu), then:

Runs on GPU:
- neighbor_counts(): the convolution (conv2d) and the pad operation
- elementwise logic: comparisons, boolean masks, where(), division, etc.
- tensor reshaping/views

Runs on CPU (or causes CPU/GPU sync):
- matplotlib rendering (must use CPU numpy arrays)
- printing scalar stats often pulls a scalar to CPU:
    .to("cpu").item()

Where scatter/gather happens (and why it matters)
-------------------------------------------------
The movement step involves irregular indexing:

- moving = flat[unhappy_idx[:k]]       (gather)
- flat[unhappy_idx[:k]] = EMPTY        (scatter)
- flat[empty_idx[:k]] = moving         (scatter)

This is fundamentally less “regular” than convolution and can become the bottleneck at scale.
If you want maximum GPU efficiency, common strategies include:
- limiting movement to local regions (domain decomposition)
- batched relocation with structured data layouts
- alternate formulations where agents are represented as fields rather than moving entities
- running many independent replicas (ensemble) instead of one huge globally-moving system

What the printed stats mean
---------------------------
- frac_unhappy: fraction of currently unhappy agents (based on similarity < threshold)
- avg_sim: mean similarity among occupied cells (a global mixing/segregation proxy)
- A/B/empty counts: sanity checks; these should remain constant (A/B) unless you change rules

Outputs produced by this script
-------------------------------
- schelling_final.png : final grid visualization saved to disk
- console logs : step-by-step progress summary (every log_every steps)

Typical usage
-------------
- For performance tests: increase L and steps.
- For ABM experiments: sweep threshold, density, frac_A and compare metrics across runs.
- For headless batch runs: keep saving images and logs; avoid interactive plt.show().

Notes on boundary conditions
----------------------------
This implementation uses circular (toroidal) boundary conditions via F.pad(..., mode="circular").
If you want fixed boundaries, change padding mode and/or handle edges explicitly.

Troubleshooting
---------------
1) If you see errors about padding_mode in conv2d:
   - PyTorch functional conv2d does not accept padding_mode.
   - This code uses F.pad(..., mode="circular") followed by F.conv2d(...). That is correct.

2) If torch.xpu is not available:
   - Your PyTorch build may not include Intel GPU support.
   - Confirm torch.xpu.is_available() and that you are on an Intel Max GPU node.

3) If plots don’t display on a compute node:
   - Use plt.savefig(...) (as the script does) instead of plt.show().
