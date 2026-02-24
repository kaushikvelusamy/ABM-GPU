import time
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

EMPTY, A, B = 0, 1, 2

def pick_device():
    # NVIDIA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon GPU (Metal / MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # Intel GPU Max (Aurora)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")

def init_grid(L=128, density=0.9, frac_A=0.5, seed=0, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    n = L * L
    n_agents = int(density * n)
    n_A = int(frac_A * n_agents)
    n_B = n_agents - n_A

    grid = torch.zeros(n, dtype=torch.int8, device=device)
    idx = torch.randperm(n, generator=g, device=device)[:n_agents]
    vals = torch.tensor([A] * n_A + [B] * n_B, dtype=torch.int8, device=device)
    vals = vals[torch.randperm(vals.numel(), generator=g, device=device)]
    grid[idx] = vals
    return grid.view(L, L)

def neighbor_counts(grid):
    device = grid.device
    A_mask = (grid == A).to(torch.float32)[None, None]  # (1,1,L,L)
    B_mask = (grid == B).to(torch.float32)[None, None]

    k = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
    k[0, 0, 1, 1] = 0.0  # don't count self

    A_pad = F.pad(A_mask, (1, 1, 1, 1), mode="circular")
    B_pad = F.pad(B_mask, (1, 1, 1, 1), mode="circular")

    A_nb = F.conv2d(A_pad, k, padding=0)[0, 0]
    B_nb = F.conv2d(B_pad, k, padding=0)[0, 0]
    return A_nb.to(torch.int16), B_nb.to(torch.int16)

@torch.no_grad()
def step_schelling(grid, threshold=0.6, seed=0):
    """
    Moves unhappy agents to random empty spots (within the whole grid).
    Returns: (grid, frac_unhappy, avg_similarity_among_occupied)
    """
    device = grid.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    A_nb, B_nb = neighbor_counts(grid)
    occ = (grid != EMPTY)
    total = (A_nb + B_nb).clamp_min(1)

    similar = torch.where(grid == A, A_nb, torch.where(grid == B, B_nb, torch.zeros_like(A_nb)))
    similarity = similar.to(torch.float32) / total.to(torch.float32)

    # avg similarity among occupied cells (just a useful stat)
    if occ.any():
        avg_sim = float(similarity[occ].mean().to("cpu").item())
    else:
        avg_sim = 0.0

    unhappy = occ & (similarity < threshold)
    unhappy_idx = unhappy.view(-1).nonzero(as_tuple=False).view(-1)
    empty_idx = (grid == EMPTY).view(-1).nonzero(as_tuple=False).view(-1)

    if unhappy_idx.numel() == 0 or empty_idx.numel() == 0:
        frac_unhappy = 0.0
        return grid, frac_unhappy, avg_sim

    unhappy_idx = unhappy_idx[torch.randperm(unhappy_idx.numel(), generator=g, device=device)]
    empty_idx = empty_idx[torch.randperm(empty_idx.numel(), generator=g, device=device)]
    k = min(unhappy_idx.numel(), empty_idx.numel())

    flat = grid.view(-1)
    moving = flat[unhappy_idx[:k]].clone()
    flat[unhappy_idx[:k]] = EMPTY
    flat[empty_idx[:k]] = moving

    frac_unhappy = unhappy_idx.numel() / max(1, int(occ.sum().to("cpu").item()))
    return grid, float(frac_unhappy), avg_sim

def run_schelling(L=128, steps=100, density=0.9, frac_A=0.5, threshold=0.6,
                  seed=0, device=None, log_every=10):
    if device is None:
        device = pick_device()
    print("Device:", device)

    grid = init_grid(L=L, density=density, frac_A=frac_A, seed=seed, device=device)

    t0 = time.time()
    for t in range(steps):
        grid, fu, avg_sim = step_schelling(grid, threshold=threshold, seed=seed + t + 1)

        if (t + 1) % log_every == 0 or t == 0:
            nA = int((grid == A).sum().to("cpu").item())
            nB = int((grid == B).sum().to("cpu").item())
            nE = int((grid == EMPTY).sum().to("cpu").item())
            dt = time.time() - t0
            print(f"step {t+1:4d}/{steps} | frac_unhappy={fu:.3f} | avg_sim={avg_sim:.3f} "
                  f"| A={nA} B={nB} empty={nE} | elapsed={dt:.2f}s")
    return grid

def plot_schelling(grid, title="Schelling ABM"):
    # 0=empty, 1=A, 2=B
    cmap = ListedColormap(["white", "tab:blue", "tab:orange"])
    arr = grid.to("cpu").numpy()

    plt.figure()
    plt.imshow(arr, interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
    nA = int((grid == A).sum().to("cpu").item())
    nB = int((grid == B).sum().to("cpu").item())
    plt.title(f"{title} | A={nA} B={nB}")
    plt.axis("off")
    # plt.show()
    plt.savefig("schelling_final.png", dpi=200, bbox_inches="tight")
    print("saved schelling_final.png")

def animate_schelling(L=128, steps=200, density=0.9, frac_A=0.5, threshold=0.6,
                      seed=0, device=None, interval_ms=50, gif_path=None, fps=20):
    if device is None:
        device = pick_device()
    print("Device:", device)

    grid = init_grid(L=L, density=density, frac_A=frac_A, seed=seed, device=device)
    cmap = ListedColormap(["white", "tab:blue", "tab:orange"])

    fig, ax = plt.subplots()
    img = ax.imshow(grid.to("cpu").numpy(), interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
    ax.axis("off")

    def update(i):
        nonlocal grid
        grid, fu, avg_sim = step_schelling(grid, threshold=threshold, seed=seed + i + 1)
        img.set_data(grid.to("cpu").numpy())
        ax.set_title(f"step={i+1}  unhappy={fu:.3f}  avg_sim={avg_sim:.3f}")
        return (img,)

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=True)
    if gif_path:
        ani.save(gif_path, writer="pillow", fps=fps)
        print(f"saved {gif_path}")
    return ani

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schelling ABM (GPU/CPU) runner")
    parser.add_argument("--mode", choices=["run", "animate"], default="run")
    parser.add_argument("--L", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--density", type=float, default=0.92)
    parser.add_argument("--frac-A", dest="frac_A", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--interval-ms", type=int, default=30)
    parser.add_argument("--gif", type=str, default="schelling.gif",
                        help="Output GIF path used in animate mode")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--minutes", type=float, default=None,
                        help="If set in animate mode, steps = minutes * 60 * fps")
    args = parser.parse_args()

    device = pick_device()

    if args.mode == "animate":
        steps = args.steps
        if args.minutes is not None:
            steps = max(1, int(args.minutes * 60 * args.fps))
        animate_schelling(
            L=args.L,
            steps=steps,
            density=args.density,
            frac_A=args.frac_A,
            threshold=args.threshold,
            seed=args.seed,
            device=device,
            interval_ms=args.interval_ms,
            gif_path=args.gif,
            fps=args.fps,
        )
    else:
        final = run_schelling(
            L=args.L,
            steps=args.steps,
            density=args.density,
            frac_A=args.frac_A,
            threshold=args.threshold,
            seed=args.seed,
            device=device,
            log_every=args.log_every,
        )
        plot_schelling(final, title="Final Schelling")
