# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Gaussian Mixture operations for GMFlow.
#
# This is a small, 1D-spatial (D = 2 for the 2D toy notebook), batched
# port of the operators in
#     https://github.com/Lakonik/GMFlow/blob/main/lib/ops/gmflow_ops/gmflow_ops.py
# Sufficient for the 2D-toy GM-SDE / GM-ODE solvers (see :mod:`solver`).
#
# A Gaussian mixture is represented as a dict
#
#     gm = {
#         "means":      [B, K, D],   # K component means
#         "logstds":    [B, 1],      # one shared scalar log-std (broadcast over D)
#         "logweights": [B, K],      # log-softmaxed weights
#     }
#
# All ops accept and return dicts of the same shape.

from typing import Optional

import math
import torch
from torch import Tensor


# ----------------------------------------------------------------------------
# Reductions
# ----------------------------------------------------------------------------


def gm_to_mean(gm: dict) -> Tensor:
    """Mean of the Gaussian mixture, ``E_{u ~ gm}[u]``.

    Returns shape ``[B, D]``.
    """
    weights = gm["logweights"].softmax(dim=-1)  # [B, K]
    return (weights.unsqueeze(-1) * gm["means"]).sum(dim=-2)


def gm_to_iso_gaussian(gm: dict) -> dict:
    """Approximate a GM with an isotropic Gaussian by matching the first two
    moments. Returns dict with ``mean: [B, D]`` and ``var: [B, 1]`` (one shared
    scalar variance).
    """
    weights = gm["logweights"].softmax(dim=-1)  # [B, K]
    g_mean = (weights.unsqueeze(-1) * gm["means"]).sum(dim=-2)  # [B, D]
    diffs = gm["means"] - g_mean.unsqueeze(-2)  # [B, K, D]
    inter_var = (weights.unsqueeze(-1) * diffs.square()).sum(dim=-2).mean(
        dim=-1, keepdim=True
    )  # [B, 1]
    intra_var = (2 * gm["logstds"]).exp()  # [B, 1]
    return {"mean": g_mean, "var": inter_var + intra_var}


def gm_to_sample(gm: dict, generator: Optional[torch.Generator] = None) -> Tensor:
    """Draw one sample per batch element from the GM.

    Returns shape ``[B, D]``.
    """
    weights = gm["logweights"].softmax(dim=-1)  # [B, K]
    B, K, D = gm["means"].shape
    idx = torch.multinomial(weights.clamp(min=1e-12), 1, generator=generator).squeeze(-1)  # [B]
    means_sel = gm["means"][torch.arange(B, device=idx.device), idx]  # [B, D]
    std = gm["logstds"].exp()  # [B, 1]
    noise = torch.randn(B, D, dtype=means_sel.dtype, device=means_sel.device, generator=generator)
    return means_sel + std * noise


# ----------------------------------------------------------------------------
# Multiplications (conflations) — used by 2nd-order GM solver
# ----------------------------------------------------------------------------


def iso_gaussian_mul_iso_gaussian(g1: dict, g2: dict, p1: float = 1.0, p2: float = 1.0, eps: float = 1e-6) -> dict:
    """Powered conflation of two isotropic Gaussians.

    ``out ∝ g1^p1 * g2^p2``. Both ``mean`` are ``[B, D]`` and both ``var``
    are ``[B, 1]``.
    """
    norm = (p1 * g2["var"] + p2 * g1["var"]).clamp(min=eps)
    out_var = g1["var"] * g2["var"] / norm
    out_mean = (p1 * g2["var"] * g1["mean"] + p2 * g1["var"] * g2["mean"]) / norm
    return {"mean": out_mean, "var": out_var}


def gm_mul_iso_gaussian(gm: dict, g: dict, gm_power: float = 1.0, gauss_power: float = 1.0, eps: float = 1e-6) -> dict:
    """Powered conflation of a GM and an isotropic Gaussian.

    ``out ∝ gm^gm_power * g^gauss_power``. Returns a new GM.
    """
    gm_means = gm["means"]                          # [B, K, D]
    gm_var = (2 * gm["logstds"]).exp().unsqueeze(-1)  # [B, 1, 1]
    g_mean = g["mean"].unsqueeze(-2)                # [B, 1, D]
    g_var = g["var"].unsqueeze(-1)                  # [B, 1, 1]

    diffs = gm_means - g_mean                       # [B, K, D]
    power_ratio = gauss_power / gm_power
    norm = (g_var + power_ratio * gm_var).clamp(min=eps)

    out_means = (g_var * gm_means + power_ratio * gm_var * g_mean) / norm
    logweights_delta = diffs.square().sum(dim=-1) * (-0.5 * power_ratio / norm.squeeze(-1))  # [B, K]
    out_logweights = torch.log_softmax(gm["logweights"] + logweights_delta, dim=-1)
    out_logstds = gm["logstds"] + 0.5 * (g["var"].clamp(min=eps).log() - norm.squeeze(-1).log())
    return {"means": out_means, "logstds": out_logstds, "logweights": out_logweights}


# ----------------------------------------------------------------------------
# Diffusion-time conversions
# ----------------------------------------------------------------------------


def u_to_x0_gm(gm_u: dict, x_t: Tensor, sigma: float, eps: float = 1e-6) -> dict:
    """Reparameterise a velocity-space GM into an x_0-space GM.

    Velocity is ``u = (x_t - x_0) / sigma`` (rectified-flow convention with
    ``alpha_t = 1 - t``, ``sigma_t = t``), so

        x_0 = x_t - sigma * u

    and the per-component std in x_0 space is ``sigma`` times the velocity std.
    """
    means_x0 = x_t.unsqueeze(-2) - sigma * gm_u["means"]  # [B, K, D]
    logstds_x0 = gm_u["logstds"] + math.log(max(sigma, eps))
    return {"means": means_x0, "logstds": logstds_x0, "logweights": gm_u["logweights"]}


def reverse_transition_gm(
    gm_u: dict,
    x_t: Tensor,
    sigma_high: float,
    sigma_low: float,
    eps: float = 1e-6,
) -> dict:
    """Analytic GM for the reverse transition ``q_θ(x_{t-Δt} | x_t)``
    (paper eq. (9)). Input GM is in velocity space.

    Returns a GM with the same component count but means/logstds describing
    a distribution over ``x_{t-Δt}``.
    """
    sigma_high_eps = max(sigma_high, eps)
    sigma_low_eps = max(sigma_low, eps)
    alpha_high = 1.0 - sigma_high
    alpha_low = 1.0 - sigma_low

    sl_over_sh = sigma_low / sigma_high_eps
    ah_over_al = alpha_high / max(alpha_low, eps)
    beta_over_sh_sq = 1.0 - (sl_over_sh * ah_over_al) ** 2  # = beta_{t,Δt} / sigma_t^2

    c1 = sl_over_sh ** 2 * ah_over_al
    c2 = beta_over_sh_sq * alpha_low
    c3 = beta_over_sh_sq * sigma_low ** 2

    # GM in x_0 space first.
    means_x0 = x_t.unsqueeze(-2) - sigma_high * gm_u["means"]            # [B, K, D]
    var_x0 = (2 * gm_u["logstds"]).exp() * (sigma_high ** 2)              # [B, 1]

    # x_{t_low} = c1 * x_t + c2 * x_0  +  sqrt(c3) * noise
    means_t_low = c1 * x_t.unsqueeze(-2) + c2 * means_x0                  # [B, K, D]
    var_t_low = (c2 ** 2) * var_x0 + c3                                  # [B, 1]
    var_t_low = var_t_low.clamp(min=eps)
    logstds_t_low = 0.5 * var_t_low.log()
    return {
        "means": means_t_low,
        "logstds": logstds_t_low,
        "logweights": gm_u["logweights"],
    }


def denoising_gm_convert_to_mean(
    gm_x0: dict,
    x_t_tgt: Tensor,
    x_t_src: Tensor,
    sigma_tgt: float,
    sigma_src: float,
    eps: float = 1e-6,
) -> Tensor:
    """Given an x_0-based GM at ``(x_t_src, sigma_src)``, derive the mean of
    the *converted* x_0-based GM at ``(x_t_tgt, sigma_tgt)`` via paper eq. (10):

        q̂(x_0 | x_τ) ∝ p(x_τ | x_0) / p(x_t | x_0) · q_θ(x_0 | x_t)

    Both Gaussians on the rhs are functions of x_0, so the ratio is itself a
    Gaussian factor. We multiply the GM by that Gaussian and return the mean
    of the resulting GM. Used by GM-ODE sub-steps and the 2nd-order solver.
    """
    alpha_tgt = 1.0 - sigma_tgt
    alpha_src = 1.0 - sigma_src
    a_tgt_s_src = alpha_tgt * sigma_src
    a_src_s_tgt = alpha_src * sigma_tgt
    denom = (a_tgt_s_src ** 2 - a_src_s_tgt ** 2)
    if abs(denom) < eps:
        denom = eps if denom >= 0 else -eps
    g_mean = (a_tgt_s_src * sigma_src * x_t_tgt - a_src_s_tgt * sigma_tgt * x_t_src) / denom  # [B, D]
    g_var = ((sigma_tgt * sigma_src) ** 2) / denom                                            # scalar
    g_var = max(g_var, eps) if denom > 0 else g_var

    g = {"mean": g_mean, "var": torch.full_like(g_mean[..., :1], float(g_var))}
    out_gm = gm_mul_iso_gaussian(gm_x0, g, gm_power=1.0, gauss_power=1.0)
    return gm_to_mean(out_gm)
