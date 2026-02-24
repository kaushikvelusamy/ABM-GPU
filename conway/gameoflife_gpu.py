import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation

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

@torch.no_grad()
def life_step(state01):
    """
    state01: (H,W) int8 or bool, values 0/1
    returns next state (H,W) int8
    """
    device = state01.device
    x = state01.to(torch.float32)[None, None]  # (1,1,H,W)

    k = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
    k[0, 0, 1, 1] = 0.0

    x_pad = F.pad(x, (1, 1, 1, 1), mode="circular")
    n = F.conv2d(x_pad, k, padding=0)[0, 0]  # (H,W) neighbor counts

    alive = state01 == 1
    new_alive = (alive & ((n == 2) | (n == 3))) | (~alive & (n == 3))
    return new_alive.to(torch.int8)

def run_life(H=256, W=256, steps=300, p=0.15, seed=0, device=None, log_every=20):
    if device is None:
        device = pick_device()
    print("Device:", device)

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    state = (torch.rand((H, W), generator=g, device=device) < p).to(torch.int8)

    t0 = time.time()
    for t in range(steps):
        state = life_step(state)

        if (t + 1) % log_every == 0 or t == 0:
            alive_count = int(state.sum().to("cpu").item())
            frac_alive = alive_count / (H * W)
            dt = time.time() - t0
            print(f"step {t+1:4d}/{steps} | alive={alive_count} ({frac_alive:.3f}) | elapsed={dt:.2f}s")

    return state

def animate_life(final_or_initial_state, steps=200, interval_ms=50, seed=0, device=None, p=None):
    """
    If you pass an initial state tensor, it will animate from that.
    If you pass a final state, it will just animate continuing from there.
    Optionally provide p to re-randomize from scratch.
    """
    if device is None:
        device = pick_device()
    print("Device:", device)

    if p is not None:
        H, W = final_or_initial_state
        g = torch.Generator(device=device); g.manual_seed(seed)
        state = (torch.rand((H, W), generator=g, device=device) < p).to(torch.int8)
    else:
        state = final_or_initial_state.to(device=device)

    fig, ax = plt.subplots()
    img = ax.imshow(state.to("cpu").numpy(), interpolation="nearest")
    ax.set_title("Game of Life (0/1)")

    def update(_):
        nonlocal state
        state = life_step(state)
        img.set_data(state.to("cpu").numpy())
        ax.set_xlabel(f"alive={int(state.sum().to('cpu').item())}")
        return (img,)

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=True)
    plt.show()
    return ani

if __name__ == "__main__":
    # 1) Run with prints and show a final plot
    device = pick_device()
    final_state = run_life(H=256, W=256, steps=200, p=0.20, seed=0, device=device, log_every=25)

    plt.figure()
    plt.imshow(final_state.to("cpu").numpy(), interpolation="nearest")
    alive = int(final_state.sum().to("cpu").item())
    plt.title(f"Final state | alive={alive}")
    # plt.show()
    plt.imsave("life_final.png", final_state.to("cpu").numpy())
    print("saved life_final.png")

    # 2) Or animate (comment out above plot if you want)
    # animate_life(final_state, steps=300, interval_ms=30, device=device)