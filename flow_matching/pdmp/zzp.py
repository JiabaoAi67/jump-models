# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Zig-Zag Process (ZZP) building blocks for piecewise deterministic generative
# models, following:
#
#     Bertazzi, Shariatian, Simsekli, Moulines, Durmus.
#     "Piecewise deterministic generative models." NeurIPS 2024.
#     arXiv:2407.19448
#
# Specialized to a standard normal target pi = N(0, I), so the per-coordinate
# forward jump rate is
#
#     lambda_i^Z(x, v) = (v_i * x_i)_+  +  lambda_r,
#
# any event flips v_i (a sign flip), and the deterministic flow is constant
# velocity dx_t = v dt, dv_t = 0.

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


# ----------------------------------------------------------------------------
# Forward simulation (paper Appendix C.1)
# ----------------------------------------------------------------------------


@torch.no_grad()
def zzp_forward(
    x_init: Tensor,
    v_init: Tensor,
    tau: Tensor,
    lambda_r: float,
    max_iters: int = 200,
) -> Tuple[Tensor, Tensor]:
    """Exactly simulate the forward ZZP from time 0 up to a per-sample time
    ``tau``, with stationary target pi = N(0, I).

    Each coordinate is simulated independently (the rate factorises because
    psi(x) = |x|^2 / 2 is separable). Per coordinate, the next event time is

        tau_grad = -v*x + sqrt((v*x)_+^2 + 2 * E_grad),  E_grad ~ Exp(1)
        tau_ref  = E_ref / lambda_r,                     E_ref  ~ Exp(1)
        tau_next = min(tau_grad, tau_ref)

    and any event flips v_i. The closed-form ``tau_grad`` follows from solving
    int_0^tau (v*x + s)_+ ds = E_grad along the deterministic flow x_s =
    x + v*s (using v in {-1, +1} so v^2 = 1).

    Args:
        x_init: ``[B, d]`` start positions, sampled from data.
        v_init: ``[B, d]`` start velocities in {-1, +1}.
        tau:    ``[B]`` per-sample target horizon in [0, T_f].
        lambda_r: refreshment rate (constant scalar).
        max_iters: hard cap on the per-coordinate event-loop length to avoid
            ever spinning forever; for the standard normal target the
            expected number of events per coord is ~2 * tau, so a few hundred
            is far more than enough.

    Returns:
        ``(x_tau, v_tau)`` both of shape ``[B, d]``.
    """
    if x_init.shape != v_init.shape:
        raise ValueError("x_init and v_init must have the same shape")
    if tau.shape[0] != x_init.shape[0]:
        raise ValueError("tau must have shape [B] matching x_init")

    B, d = x_init.shape
    x = x_init.clone()
    v = v_init.clone().to(x.dtype)
    t_local = torch.zeros_like(x)                    # current local time per coord
    T = tau.to(x.dtype)[:, None].expand(-1, d).contiguous()

    eps = 1e-12
    for _ in range(max_iters):
        active = t_local < T - 1e-9
        if not bool(active.any()):
            break

        # Sample two competing event times for every coord (cheap; mask later).
        u1 = torch.rand_like(x).clamp_(min=eps)
        E_grad = -torch.log(u1)
        vx = v * x
        vx_pos_sq = vx.clamp(min=0.0).square()
        tau_grad = -vx + torch.sqrt(vx_pos_sq + 2.0 * E_grad)

        if lambda_r > 0:
            u2 = torch.rand_like(x).clamp_(min=eps)
            tau_ref = -torch.log(u2) / lambda_r
            tau_next = torch.minimum(tau_grad, tau_ref)
        else:
            tau_next = tau_grad

        # Time after the proposed event.
        new_t = t_local + tau_next
        will_complete = new_t >= T

        # Effective dt this iteration: full event step, or shrink to T.
        dt = torch.where(will_complete, T - t_local, tau_next)
        # Mask out coords that already finished.
        dt = torch.where(active, dt, torch.zeros_like(dt))

        # Move along the deterministic flow.
        x = x + v * dt
        # Flip velocity only on real events (active and not yet complete).
        flip = active & (~will_complete)
        v = torch.where(flip, -v, v)
        t_local = t_local + dt

    return x, v


# ----------------------------------------------------------------------------
# Implicit ratio matching loss, simplified model variant (paper Appendix D.1,
# eq. (24))
# ----------------------------------------------------------------------------


def zzp_simple_ratio_loss(
    s_plus: Tensor,
    s_minus: Tensor,
    v: Tensor,
) -> Tensor:
    r"""Simplified implicit ratio matching loss for ZZP.

    For each coordinate i,

        l_i = G^2(s_curr) + G^2(s_flip) - 2 * G(s_curr),     G(r) = 1 / (1 + r),

    where ``s_curr = s_{sign(v_i), i}`` is the model's estimate of
    ``p_t(-v_i | x) / p_t(v_i | x)`` (the ratio for flipping coord i out of
    its current sign), and ``s_flip = s_{-sign(v_i), i}`` is the symmetric
    estimate.

    The minimum corresponds to the true conditional density ratios
    (paper Proposition 3 + Appendix D.1).

    Args:
        s_plus:  ``[B, d]`` model output for v_i = +1 (positive).
        s_minus: ``[B, d]`` model output for v_i = -1 (positive).
        v:       ``[B, d]`` velocities in {-1, +1}.

    Returns:
        Scalar loss (mean over batch, sum over coords).
    """
    if s_plus.shape != s_minus.shape:
        raise ValueError("s_plus and s_minus must have the same shape")
    if v.shape != s_plus.shape:
        raise ValueError("v must match s_plus shape")

    is_plus = (v > 0).to(s_plus.dtype)
    s_curr = is_plus * s_plus + (1.0 - is_plus) * s_minus
    s_flip = is_plus * s_minus + (1.0 - is_plus) * s_plus

    G_curr = 1.0 / (1.0 + s_curr)
    G_flip = 1.0 / (1.0 + s_flip)
    loss_per_coord = G_curr.square() + G_flip.square() - 2.0 * G_curr
    return loss_per_coord.sum(dim=-1).mean()


# ----------------------------------------------------------------------------
# Backward DJD splitting sampler (paper Algorithm 3)
# ----------------------------------------------------------------------------


@torch.no_grad()
def zzp_djd_sample(
    model: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
    n_samples: int,
    T_f: float,
    lambda_r: float,
    n_steps: int,
    device: torch.device,
    d: int = 2,
    return_trajectory: bool = False,
    quadratic_schedule: bool = True,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""Backward sampler for the time-reversed ZZP using the DJD splitting
    scheme of Bertazzi et al. [2023] (paper Algorithm 3, simplified model).

    Each backward step is a half-step of reversed deterministic motion, then
    a per-coordinate flip with probability
    ``1 - exp(- delta * s_curr * lambda_i(x, R_i v))``, then another half-step
    of reversed deterministic motion. The model is queried once per backward
    step at the midpoint forward time ``T_f - t_n - delta/2``.

    Args:
        model: callable ``(x [B, d], t [B]) -> (s_plus [B, d], s_minus [B, d])``
            returning positive density-ratio estimates.
        n_samples: batch size.
        T_f: forward horizon used during training.
        lambda_r: refreshment rate used during training.
        n_steps: number of backward steps. Total NFE equals ``n_steps``.
        device: torch device.
        d: state dimension (default 2 for the toy experiments).
        return_trajectory: if True, also return the position trajectory of
            shape ``[n_steps + 1, B, d]``.
        quadratic_schedule: if True, use the quadratic time grid recommended
            in paper Appendix F. Otherwise use a uniform grid.

    Returns:
        ``(x_final, v_final, traj_or_None)``. ``traj_or_None`` is ``None`` if
        ``return_trajectory`` is False, else a CPU tensor of intermediate
        positions.
    """
    # Initialise from pi (x) ⊗ ν (v): standard normal positions, uniform
    # +/-1 velocities. (See paper Section 2.1.)
    x = torch.randn(n_samples, d, device=device)
    v = (
        2.0
        * torch.randint(0, 2, (n_samples, d), device=device).to(x.dtype)
        - 1.0
    )

    # Time grid for the *backward* run, in original (forward) time units.
    s = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if quadratic_schedule:
        t_grid = T_f * s.square()
    else:
        t_grid = T_f * s
    deltas = (t_grid[1:] - t_grid[:-1]).cpu().tolist()
    t_grid_cpu = t_grid.cpu().tolist()

    traj = [x.detach().cpu().clone()] if return_trajectory else None

    for n in range(n_steps):
        delta = deltas[n]
        t_n = t_grid_cpu[n]

        # Half-step backward deterministic motion: x <- x - (delta/2) * v.
        x = x - 0.5 * delta * v

        # Midpoint forward time used to query the model.
        t_tilde = T_f - t_n - 0.5 * delta
        t_in = torch.full((n_samples,), t_tilde, device=device, dtype=x.dtype)
        s_plus, s_minus = model(x, t_in)

        # Pick the ratio that corresponds to the current velocity sign.
        is_plus = (v > 0).to(s_plus.dtype)
        s_curr = is_plus * s_plus + (1.0 - is_plus) * s_minus

        # Forward jump rate evaluated at the *flipped* state, see Prop. 2(1):
        #   lambda_i(x, R_i v) = ((-v_i) * x_i)_+ + lambda_r.
        rate_flipped = (-v * x).clamp(min=0.0) + lambda_r

        # Backward jump rate (paper eq. (4)).
        bw_rate = s_curr * rate_flipped

        # Per-coord flip with probability 1 - exp(-delta * bw_rate).
        p_flip = 1.0 - torch.exp(-delta * bw_rate)
        flip = torch.rand_like(p_flip) < p_flip
        v = torch.where(flip, -v, v)

        # Second half-step of reversed deterministic motion.
        x = x - 0.5 * delta * v

        if return_trajectory:
            traj.append(x.detach().cpu().clone())

    traj_tensor = torch.stack(traj, dim=0) if return_trajectory else None
    return x, v, traj_tensor
