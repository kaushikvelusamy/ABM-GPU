Game of Life on CPU / NVIDIA GPU / Intel Data Center GPU Max (PyTorch)

What this code does

This script runs Conway’s Game of Life, a classic cellular automaton on a 2D grid.
	•	The world is an H × W grid.
	•	Each cell is either:
	•	0 = dead
	•	1 = alive
	•	The system evolves in discrete time steps.
	•	At each step, every cell looks at its 8 neighbors (Moore neighborhood) and updates simultaneously using Life’s rules.

Life rules (what is being modeled)

Let n be the number of alive neighbors for a cell:
	•	If the cell is alive:
	•	it stays alive if n is 2 or 3
	•	otherwise it dies (underpopulation / overcrowding)
	•	If the cell is dead:
	•	it becomes alive if n is exactly 3 (birth)

This model is often used as a minimal demonstration of emergence: complex global patterns can form from simple local rules (oscillators, gliders, stable structures, etc.).

⸻

Why this is a “GPU-friendly” problem

Game of Life is basically a repeated neighborhood-sum + rule-application:
	1.	For each cell, compute n = (# of alive neighbors)
	2.	Apply a few boolean rules to produce the next grid

That’s a “stencil” computation: the same small pattern of work repeated over every cell. GPUs love this because:
	•	The computation is regular
	•	The memory access pattern is predictable
	•	Every cell can be processed in parallel

⸻

What is PyTorch doing here?

Even though this is not a neural network, PyTorch is useful because it provides:
	1.	Device-aware arrays (“tensors”) that can live on CPU or GPU
	2.	Fast, optimized implementations of common building blocks, including:
	•	2D convolution (conv2d), which is exactly a fast way to compute neighbor sums

Think of PyTorch here as:
	•	a “high-performance array library”
	•	plus a large set of optimized kernels that run on CPU/GPU depending on where the data lives

⸻

What does “3×3 convolution” mean in this code?

The key operation in Game of Life is counting neighbors.

Neighbor count as a 3×3 sliding window

Imagine placing a 3×3 window on the grid centered at each cell:
a b c
d X e
f g h
To count neighbors, you sum the 8 surrounding values (a,b,c,d,e,f,g,h) but not the center cell X.

This code uses a 3×3 kernel:
1 1 1
1 0 1
1 1 1

When you convolve the grid with this kernel, each output cell becomes the sum of its 8 neighbors.

So:
	•	F.pad(..., mode="circular") makes the grid wrap around at edges (toroidal world)
	•	F.conv2d(...) computes the neighbor counts for every cell efficiently

⸻

What runs on the GPU vs CPU?

Device selection

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")  # Intel Data Center GPU Max
    return torch.device("cpu")

    	•	cuda = NVIDIA GPU
	•	xpu = Intel GPU (Data Center GPU Max)
	•	otherwise cpu

The rule of thumb

If state is on the GPU, PyTorch runs operations on the GPU.

In this script, the heavy work per step is:
	•	F.conv2d(...) neighbor counting
	•	boolean comparisons (n == 3), (n == 2), etc.

Those execute on GPU when the tensors are on cuda or xpu.

The CPU parts
	•	Printing stats requires retrieving a number to CPU:
alive_count = int(state.sum().to("cpu").item())
That forces a small synchronization. It’s fine because it only happens every log_every steps.

	•	Plotting/animation uses matplotlib, which is CPU-side. We convert the tensor to CPU for display:

state.to("cpu").numpy()

What “problem” this code is solving (in practical terms)

For an ABM/CA expert, the model is simple; the interesting part is the implementation strategy:
	•	“How do I run a grid-based local-interaction model efficiently on an accelerator (Intel Max / NVIDIA)?”
	•	“How do I avoid writing custom GPU kernels (SYCL/CUDA) for a stencil update?”

This code demonstrates a very practical pattern:

Local neighborhood computations → express as convolution → let a mature kernel library handle acceleration.

That pattern carries over to many ABM-like grid models:
	•	forest fire
	•	reaction–diffusion
	•	epidemic CA
	•	lattice gas / Ising-like updates (with modifications)
	•	agent density fields / local influence maps

⸻

How to run

Run simulation and save final image

The main block does:
	•	selects device
	•	runs for steps
	•	prints alive fraction periodically
	•	saves life_final.png

Output example:

Device: xpu
step    1/200 | alive=13212 (0.201) | elapsed=0.05s
step   25/200 | alive= 8123 (0.124) | elapsed=0.32s
...
saved life_final.png

Animation

Uncomment:
# animate_life(final_state, steps=300, interval_ms=30, device=device)

Note: animation requires a display. On headless compute nodes, prefer saving images.

⸻

Notes and tuning
	•	H, W: larger grids increase GPU utilization and often improve throughput.
	•	steps: more steps = longer runtime, but better amortizes overhead.
	•	log_every: set higher to reduce CPU sync overhead from printing.
	•	mode="circular" padding makes the grid wrap around; if you want fixed boundaries, use constant padding instead.

