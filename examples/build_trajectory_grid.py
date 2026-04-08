"""Build a 2x3 grid of particle trajectory plots, one per method.

Layout:
    Flow Matching | Jump-only        | Jump + Flow
    GMFlow        | PDGM-ZZP         | DLPM (alpha=1.8)

Each panel:
- 3 backward trajectories from the same fixed starting points (NFE=100)
- For methods with discrete-event dynamics, jump-step markers (black "x")

Jump-step definitions:
- Jump-only / Jump+Flow: per-step Bernoulli jump event (existing logic).
- PDGM-ZZP: per-step velocity flip (any v_i changes sign).
- DLPM: per-step heavy-tail event -- max over the per-coord auxiliary
    a_t exceeds 5. At alpha=1.8 the per-coord a has median ~1.76 and
    q99 ~6, so a > 5 picks out the upper ~5% mass: noise amplitude
    sqrt(a) > sqrt(5) ~ 2.24, which is 1.6x the Gaussian baseline
    sqrt(2). Physically: "heavy-tail multiplier so big a Gaussian
    sampler would basically never see it".
- Flow / GMFlow: no jump markers (deterministic ODE / Gaussian SDE).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from flow_matching.gmflow import GMFlowSolver
from flow_matching.dlpm import DLPMSchedule, sample_skewed_levy


# ============================================================================
# config
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

CKPT_DIR = "/home/jiabao/workspace/flow_matching/examples"
ASSET_DIR = "/home/jiabao/workspace/flow_matching/assets"
os.makedirs(ASSET_DIR, exist_ok=True)

HIDDEN_DIM = 512
NUM_BINS = 64
DATA_RANGE = 5.5
NUM_GAUSSIANS = 8
T_F = 5.0
LAMBDA_R = 1.0
DLPM_ALPHA = 1.8
DLPM_DATA_SCALE = 1.0
DLPM_JUMP_A_THRESH = 5.0   # ~5% of steps at alpha=1.8 (q95 of max(a, axis=d=2))

NFE = 100
N_TRAJ = 3
SEED = 0


# ============================================================================
# architectures (copied verbatim from build_combined_gif.py)
# ============================================================================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FlowMLP(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, 2),
        )
    def forward(self, x, t):
        t_in = t.reshape(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_in], dim=1))


class JumpMLP(nn.Module):
    def __init__(self, hidden_dim=512, num_bins=64, with_flow=False):
        super().__init__()
        self.num_bins = num_bins
        self.with_flow = with_flow
        self.backbone = nn.Sequential(
            nn.Linear(3, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
        )
        self.jump_logits_head = nn.Linear(hidden_dim, 2 * num_bins)
        self.log_lambda_head = nn.Linear(hidden_dim, 2)
        if with_flow:
            self.velocity_head = nn.Linear(hidden_dim, 2)
    def forward(self, x, t):
        B = x.shape[0]
        t_in = t.reshape(-1, 1).expand(B, 1)
        feat = self.backbone(torch.cat([x, t_in], dim=1))
        logits = self.jump_logits_head(feat).view(B, 2, self.num_bins)
        log_lambda = self.log_lambda_head(feat)
        if self.with_flow:
            return self.velocity_head(feat), logits, log_lambda
        return logits, log_lambda


class GMFlowMLP(nn.Module):
    def __init__(self, hidden_dim=512, num_gaussians=8):
        super().__init__()
        self.K = num_gaussians
        self.backbone = nn.Sequential(
            nn.Linear(3, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
        )
        self.out_means = nn.Linear(hidden_dim, num_gaussians * 2)
        self.out_logweights = nn.Linear(hidden_dim, num_gaussians)
        self.out_logstd = nn.Linear(hidden_dim, 1)
    def forward(self, x, t):
        B = x.shape[0]
        t_in = t.reshape(-1, 1).expand(B, 1)
        feat = self.backbone(torch.cat([x, t_in], dim=1))
        means = self.out_means(feat).reshape(B, self.K, 2)
        logweights = torch.log_softmax(self.out_logweights(feat), dim=-1)
        logstds = self.out_logstd(feat)
        return dict(means=means, logstds=logstds, logweights=logweights)


class ZZPMLP(nn.Module):
    def __init__(self, hidden_dim=512, T_f=5.0):
        super().__init__()
        self.T_f = T_f
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, 4),
        )
    def forward(self, x, t):
        t_in = (t / self.T_f).reshape(-1, 1).expand(x.shape[0], 1)
        out = F.softplus(self.net(torch.cat([x, t_in], 1)))
        return out[:, :2], out[:, 2:]


class DLPMMLP(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, 2),
        )
    def forward(self, x, t):
        t_in = t.reshape(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_in], dim=1))


def load_state(model, name):
    path = os.path.join(CKPT_DIR, f"_{name}_ckpt.pt")
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    model.load_state_dict(obj)
    model.eval()
    return model


flow_model   = load_state(FlowMLP(HIDDEN_DIM).to(device),                              "flow")
jump_model   = load_state(JumpMLP(HIDDEN_DIM, NUM_BINS, with_flow=False).to(device),    "jump")
jf_model     = load_state(JumpMLP(HIDDEN_DIM, NUM_BINS, with_flow=True ).to(device),    "jumpflow")
gmflow_model = load_state(GMFlowMLP(HIDDEN_DIM, NUM_GAUSSIANS).to(device),              "gmflow")
zzp_model    = load_state(ZZPMLP(HIDDEN_DIM, T_F).to(device),                           "zzp")
dlpm_model   = load_state(DLPMMLP(HIDDEN_DIM).to(device),                               "dlpm")
print("All models loaded.")


# ============================================================================
# samplers (each returns (traj [n_steps+1, B, d], jump_mask [n_steps, B] or None))
# ============================================================================
bin_centers = torch.linspace(-DATA_RANGE, DATA_RANGE, NUM_BINS + 1, device=device)
bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2.0


@torch.no_grad()
def sample_flow(x_init, n_steps):
    class Wrap(ModelWrapper):
        def forward(self, x, t, **k):
            return self.model(x, t)
    solver = ODESolver(velocity_model=Wrap(flow_model))
    Tg = torch.linspace(0, 1, n_steps + 1).to(device)
    sol = solver.sample(time_grid=Tg, x_init=x_init, method="midpoint",
                        step_size=1.0/n_steps, return_intermediates=True)
    return sol.cpu(), None


@torch.no_grad()
def sample_jump_traj(model, x_init, n_steps, with_flow, alpha):
    """Per-step Bernoulli jump (existing logic from the notebook)."""
    bc = bin_centers
    nb = bc.shape[0]
    bin_width = (bc[-1] - bc[0]).item() / (nb - 1)
    t_grid = torch.linspace(0.0, 1.0, n_steps + 1).to(device)
    x_t = x_init.clone()
    traj = [x_t.cpu()]
    jumped = []
    for i in range(n_steps):
        t_val = t_grid[i].item()
        h = t_grid[i + 1].item() - t_val
        t_tensor = torch.full((x_t.shape[0],), t_val, device=device)
        if with_flow:
            velocity, jump_logits, log_lambda = model(x_t, t_tensor)
        else:
            jump_logits, log_lambda = model(x_t, t_tensor)
        lambda_t = F.softplus(log_lambda)
        B, D, _ = jump_logits.shape
        J_flat = torch.softmax(jump_logits, dim=-1).view(B * D, nb)
        bin_idx = torch.multinomial(J_flat.clamp(min=1e-8), 1).squeeze(-1).view(B, D)
        x_bin = bc[bin_idx]
        x_jump = x_bin + (torch.rand_like(x_bin) - 0.5) * bin_width
        p_jump = (alpha * h * lambda_t).clamp(max=1.0)
        mask_jump = torch.rand_like(lambda_t) < p_jump
        jumped.append(mask_jump.any(dim=-1).cpu())
        if with_flow:
            x_flow = x_t + (1.0 - alpha) * h * velocity
            x_t = torch.where(mask_jump, x_jump, x_flow)
        else:
            x_t = torch.where(mask_jump, x_jump, x_t)
        traj.append(x_t.cpu())
    return torch.stack(traj, dim=0), torch.stack(jumped, dim=0)


@torch.no_grad()
def sample_gmflow(x_init, n_steps):
    """GMFlow's GM-SDE 2 sampler -- ride-along trajectory.
    GMFlowSolver builds its own initial noise so we accept x_init only for
    parity of API; the seed controls all randomness."""
    n_samples = x_init.shape[0]
    solver = GMFlowSolver(gmflow_model, mode="sde", order=2)
    _, traj = solver.sample(n_samples=n_samples, n_steps=n_steps, device=device,
                            return_trajectory=True, seed=SEED)
    return traj, None


@torch.no_grad()
def sample_pdgm(x_init, n_steps):
    """Time-reversed Zig-Zag DJD sampler with per-step velocity-flip mask.

    Inlined copy of zzp_djd_sample so we can capture the per-step `flip`
    tensor (any v_i changing sign in this backward step).
    """
    n_samples = x_init.shape[0]
    d = x_init.shape[1]

    # Init: pi(x) ⊗ nu(v). x_init is unused for x (we sample fresh)
    # because the official sampler does so; v is fresh +/-1.
    torch.manual_seed(SEED)
    x = torch.randn(n_samples, d, device=device)
    v = 2.0 * torch.randint(0, 2, (n_samples, d), device=device).to(x.dtype) - 1.0

    s = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    t_grid = T_F * s.square()  # quadratic schedule, paper App. F
    deltas = (t_grid[1:] - t_grid[:-1]).cpu().tolist()
    t_grid_cpu = t_grid.cpu().tolist()

    traj = [x.detach().cpu().clone()]
    jumped = []

    for n in range(n_steps):
        delta = deltas[n]
        t_n = t_grid_cpu[n]
        x = x - 0.5 * delta * v
        t_tilde = T_F - t_n - 0.5 * delta
        t_in = torch.full((n_samples,), t_tilde, device=device, dtype=x.dtype)
        s_plus, s_minus = zzp_model(x, t_in)
        is_plus = (v > 0).to(s_plus.dtype)
        s_curr = is_plus * s_plus + (1.0 - is_plus) * s_minus
        rate_flipped = (-v * x).clamp(min=0.0) + LAMBDA_R
        bw_rate = s_curr * rate_flipped
        p_flip = 1.0 - torch.exp(-delta * bw_rate)
        flip = torch.rand_like(p_flip) < p_flip
        v = torch.where(flip, -v, v)
        x = x - 0.5 * delta * v
        jumped.append(flip.any(dim=-1).cpu())
        traj.append(x.detach().cpu().clone())

    return torch.stack(traj, dim=0), torch.stack(jumped, dim=0)


@torch.no_grad()
def sample_dlpm(x_init, n_steps):
    """Inlined DLPM stochastic backward sampler that also returns
    a per-step `jump_mask[n, b]` indicating whether the per-coord
    heavy-tail multiplier `a` at this step exceeded the threshold
    DLPM_JUMP_A_THRESH (5.0 at alpha=1.8 ~= upper 5% of mass).
    """
    n_samples = x_init.shape[0]
    d = x_init.shape[1]
    gen = torch.Generator(device=device).manual_seed(SEED)
    schedule = DLPMSchedule(alpha=DLPM_ALPHA, num_steps=n_steps, device=device)
    T = schedule.num_steps
    g = schedule.gammas
    s = schedule.sigmas
    bs = schedule.bar_sigmas
    eps_clamp = 1e-8

    # 1) Sample full A_{1:T}
    A = torch.stack([
        sample_skewed_levy(DLPM_ALPHA, (n_samples, d), device,
                           generator=gen, clamp_a=100.0)
        for _ in range(T)
    ], dim=0)  # [T, B, d]
    # 2) Cumulative Sigmas
    Sigmas = [s[0] ** 2 * A[0]]
    for tt in range(1, T):
        Sigmas.append(s[tt] ** 2 * A[tt] + g[tt] ** 2 * Sigmas[-1])
    Sigmas = torch.stack(Sigmas, dim=0)
    # 3) Init noise
    init_a = sample_skewed_levy(DLPM_ALPHA, (n_samples, d), device,
                                generator=gen, clamp_a=100.0)
    init_z = torch.randn(n_samples, d, device=device, generator=gen)
    x_t = bs[-1] * torch.sqrt(init_a) * init_z

    traj = [x_t.detach().cpu().clone()]
    jumped = []
    # 4) Backward loop -- t_cur = T-1, T-2, ..., 1
    for i in range(T - 1, 0, -1):
        t_cur = i
        t_in = torch.full((n_samples,), float(t_cur) / T, device=device, dtype=x_t.dtype)
        eps_pred = dlpm_model(x_t, t_in)
        Gamma_t = 1.0 - (g[t_cur] ** 2 * Sigmas[t_cur - 1]) / Sigmas[t_cur].clamp(min=eps_clamp)
        Gamma_t = Gamma_t.clamp(min=0.0)
        mean = (x_t - bs[t_cur] * Gamma_t * eps_pred) / g[t_cur].clamp(min=eps_clamp)
        var = Gamma_t * Sigmas[t_cur - 1]
        nonzero = 1.0 if t_cur != 1 else 0.0
        z = torch.randn(n_samples, d, device=device, generator=gen)
        x_t = mean + nonzero * torch.sqrt(var.clamp(min=0.0)) * z
        # Mark jump if the per-coord a at THIS step is in the heavy tail.
        a_t_max = A[t_cur].max(dim=-1).values  # [B]
        jumped.append((a_t_max > DLPM_JUMP_A_THRESH).cpu())
        traj.append(x_t.detach().cpu().clone() * DLPM_DATA_SCALE)

    # The loop produced T-1 jumped entries but T frames; pad with one extra
    # False at the front (init step has no preceding jump). Or just shift.
    # Actually we want jumped[k] to mean "jump at backward step k", aligned
    # with traj[1+k]. The loop has T-1 iterations producing T-1 entries,
    # giving traj of length T. So jumped has length T-1 == n_steps - 1.
    # That's fine -- we just align jumped[k] -> traj[1+k] in the plotter.
    return torch.stack(traj, dim=0), torch.stack(jumped, dim=0)


# ============================================================================
# Sample 3 trajectories per method, with the SAME starting points
# ============================================================================
print("\nSampling 3 trajectories per method ...")
torch.manual_seed(SEED)
x0 = torch.randn(N_TRAJ, 2, device=device)

flow_traj,    _         = sample_flow(x0, NFE);                              print("  flow OK")
jump_traj,    jump_mask = sample_jump_traj(jump_model, x0, NFE, False, 1.0);  print("  jump OK")
jf_traj,      jf_mask   = sample_jump_traj(jf_model,   x0, NFE, True,  0.5);  print("  jumpflow OK")
gmflow_traj,  _         = sample_gmflow(x0, NFE);                              print("  gmflow OK")
pdgm_traj,    pdgm_mask = sample_pdgm(x0, NFE);                                print("  pdgm OK")
dlpm_traj,    dlpm_mask = sample_dlpm(x0, NFE);                                print("  dlpm OK")

# Sanity-print a few jump counts
print(f"\nJump counts (per particle, NFE={NFE}):")
print(f"  jump:   {jump_mask.sum(0).tolist()}")
print(f"  jf:     {jf_mask.sum(0).tolist()}")
print(f"  pdgm:   {pdgm_mask.sum(0).tolist()}")
print(f"  dlpm:   {dlpm_mask.sum(0).tolist()}")


# ============================================================================
# Build the 2x3 figure
# ============================================================================
print("\nRendering 2x3 trajectory figure ...")

PANELS = [
    ("Flow Matching",    flow_traj,   None,        "tab:red"),
    ("Jump-only",        jump_traj,   jump_mask,   "tab:blue"),
    ("Jump + Flow",      jf_traj,     jf_mask,     "tab:green"),
    ("GMFlow",           gmflow_traj, None,        "tab:orange"),
    ("PDGM-ZZP",         pdgm_traj,   pdgm_mask,   "tab:purple"),
    ("DLPM (\u03b1=1.8)", dlpm_traj,  dlpm_mask,   "tab:brown"),
]

# Faint checkerboard background showing the target distribution
def inf_train_gen(B):
    x1 = torch.rand(B) * 4 - 2
    x2_ = torch.rand(B) - torch.randint(high=2, size=(B,)) * 2
    x2 = x2_ + (torch.floor(x1) % 2)
    return torch.cat([x1[:, None], x2[:, None]], 1) / 0.45

bg_pts = inf_train_gen(8000).numpy()

fig, axs = plt.subplots(2, 3, figsize=(12.5, 8.6))
fig.suptitle(
    f"Per-particle backward trajectories (NFE={NFE}, 3 fixed initial points)",
    fontsize=13, y=0.995,
)

for ax, (title, traj, mask, color) in zip(axs.flat, PANELS):
    ax.set_facecolor("white")
    ax.set_xlim(-5.2, 5.2); ax.set_ylim(-5.2, 5.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb")
    ax.set_title(title, fontsize=12)

    # faint target checkerboard so the viewer sees where particles are heading
    ax.scatter(bg_pts[:, 0], bg_pts[:, 1], s=1.0, c="#dddddd", zorder=1)

    for p in range(N_TRAJ):
        x = traj[:, p, 0].numpy()
        y = traj[:, p, 1].numpy()
        ax.plot(x, y, color=color, lw=1.4, alpha=0.85, zorder=2)
        # init dot (start)
        ax.scatter(x[0], y[0], marker="o", facecolor="white",
                   edgecolor="black", s=46, lw=1.2, zorder=4)
        # final square (end)
        ax.scatter(x[-1], y[-1], marker="s", facecolor=color,
                   edgecolor="black", s=48, lw=1.0, zorder=4)
        # jump markers (where applicable)
        if mask is not None:
            steps = mask[:, p].numpy()
            # jump_mask[k] aligns with traj[1+k]
            jx = x[1:1 + len(steps)][steps]
            jy = y[1:1 + len(steps)][steps]
            ax.scatter(jx, jy, marker="x", c="black", s=58, lw=1.4, zorder=5)

# Shared legend at the bottom
legend_elems = [
    Line2D([0], [0], marker="o", color="white", markerfacecolor="white",
           markeredgecolor="black", markersize=8, lw=0, label="start"),
    Line2D([0], [0], marker="s", color="white", markerfacecolor="grey",
           markeredgecolor="black", markersize=8, lw=0, label="end"),
    Line2D([0], [0], marker="x", color="black", markersize=9, lw=1.4,
           ls="none", label="jump event"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3,
           fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.005))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out_path = os.path.join(ASSET_DIR, "particle_trajectories.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")
