# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Denoising Levy Probabilistic Models (DLPM) - faithful port of
#     https://github.com/darioShar/DLPM/blob/main/dlpm/methods/dlpm.py
#     https://github.com/darioShar/DLPM/blob/main/dlpm/methods/GenerativeLevyProcess.py
#     https://github.com/darioShar/DLPM/blob/main/bem/datasets/torchlevy/levy.py
# Specialised to the 2D toy notebook setting.
#
# Reference:
#   Shariatian, Simsekli, Durmus.
#   "Denoising Levy Probabilistic Models." ICLR 2025.  arXiv:2407.18609
#
# DLPM is structurally a DDPM where the Gaussian noise is replaced by a
# symmetric alpha-stable (S-alpha-S) Levy noise. The forward marginal is
#
#     x_t = bar_gamma_t * x_0 + bar_sigma_t * eps,    eps ~ S-alpha-S
#
# but training does NOT directly draw the heavy-tailed `eps`. The official
# code uses the *augmented* representation `eps = sqrt(a_t) * z_t`, where
# `a_t` is positive (totally skewed) alpha/2-stable and `z_t` is standard
# Gaussian. Conditional on `a_t`, the forward step is Gaussian, so the L2
# loss on `eps_t = sqrt(a_t) * z_t` is bounded *per training sample* (the
# heavy-tailed marginal is reproduced by averaging over `a_t`). This is the
# `get_one_rv_loss_elements` path in the official code; the same path is
# what the official 2D config (`dlpm/configs/2d_data.yml`) uses
# (`mean_predict: EPSILON`, `loss_type: EPS_LOSS`, `lploss: 2.0`).
#
# Sampling here uses DLIM with eta=0 (deterministic), which is the
# alpha-stable analogue of DDIM-eta=0 and reduces to standard DDIM at
# alpha=2. The official code also has a stochastic DLPM sampler that
# precomputes a path of `a_{1:T}` r.v.'s (`p_sample_loop_progressive`); we
# do not need that machinery for the 2D toy and skip it for simplicity.

import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


# ----------------------------------------------------------------------------
# Stable distribution sampler (Chambers-Mallows-Stuck), ported from
# bem/datasets/torchlevy/levy.py :: LevyStable._sample
# ----------------------------------------------------------------------------


def _sample_stable(
    alpha_inner: float,
    beta: float,
    shape,
    device,
    dtype=torch.float64,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    r"""Chambers-Mallows-Stuck (1976) sampler for a general
    :math:`S(\alpha_\text{inner}, \beta, \mathrm{loc}{=}0, \mathrm{scale}{=}1)`
    stable distribution. Direct port of the ``otherwise`` branch in the
    official ``LevyStable._sample`` (which is what gets used for
    :math:`\alpha \neq 1`, :math:`\beta \neq 0`).

    Note: the official code samples ``TH`` from a slightly trimmed range
    :math:`(-\pi/2 + 0.15,\, \pi/2 - 0.15)` instead of the full
    :math:`(-\pi/2, \pi/2)`, presumably to avoid numerical issues near the
    boundary. We match that here.
    """
    if abs(alpha_inner - 1.0) < 1e-9 or abs(beta) < 1e-9:
        raise NotImplementedError(
            "Only the alpha != 1, beta != 0 branch is needed for DLPM"
        )

    th_range = math.pi - 0.3
    TH = (torch.rand(shape, device=device, dtype=dtype, generator=generator) * th_range
          - th_range / 2.0)
    W = -torch.log(
        torch.rand(shape, device=device, dtype=dtype, generator=generator).clamp(min=1e-30)
    )  # Exp(1)
    aTH = alpha_inner * TH
    cosTH = torch.cos(TH)
    tanTH = torch.tan(TH)

    # ``otherwise`` branch from the official _sample
    val0 = beta * math.tan(math.pi * alpha_inner / 2.0)
    th0 = math.atan(val0) / alpha_inner
    val3 = W / (cosTH / torch.tan(alpha_inner * (th0 + TH)) + torch.sin(TH))
    base = (
        (torch.cos(aTH) + torch.sin(aTH) * tanTH
         - val0 * (torch.sin(aTH) - torch.cos(aTH) * tanTH)) / W
    ).clamp(min=1e-30)
    res = val3 * base.pow(1.0 / alpha_inner)
    return res.to(torch.float32)


# ----------------------------------------------------------------------------
# gen_skewed_levy and gen_sas, mirroring torchlevy/levy.py
# ----------------------------------------------------------------------------


def sample_skewed_levy(
    alpha: float,
    shape,
    device,
    generator: Optional[torch.Generator] = None,
    clamp_a: Optional[float] = 2000.0,
) -> Tensor:
    r"""Positive (totally skewed, :math:`\beta = 1`)
    :math:`\alpha/2`-stable random variable, used as the auxiliary "a" in
    the augmented representation of S-alpha-S noise.

    Faithful port of the official ``LevyStable.sample`` ``is_isotropic``
    branch, which produces the canonical :math:`S^i_\alpha(0, \mathrm{scale}{=}1)`
    distribution via the augmented identity from paper Theorem 1:

    .. math::
        X = \sqrt{A}\, G,
        \quad A \sim S_{\alpha/2,\, 1}(0,\, 2 c_A),
        \quad G \sim \mathcal{N}(0, I_d).

    The factor of 2 comes from matching CMS scale-1 conventions (the paper
    text writes ``c_A = cos^(2/alpha)(pi alpha / 4)`` but, as the empirical
    median of ``sqrt(a)*z`` and a direct CMS draw of ``S_alpha(0, 1)`` show,
    the implementation uses ``2 * c_A`` so that the augmented form matches
    the canonical scale-1 SaS exactly).

    Crucially this means at :math:`\alpha \to 2` the prior is
    :math:`\mathcal{N}(0, 2)`, **not** :math:`\mathcal{N}(0, 1)`. So the DLPM
    init noise has noticeably wider scale than the ``torch.randn`` priors
    used by Flow Matching / Jump / GMFlow / PDGM, *and* visibly heavier
    tails for :math:`\alpha < 2`. That heavier tail is the whole point of
    DLPM (rare large "jumps" help reach isolated modes -- see paper
    Introduction), so we keep it visible rather than try to mask it.

    The optional ``clamp_a`` matches the official ``gen_skewed_levy``'s
    ``clamp_a = 2000`` default to limit pathological tail outliers.
    """
    if abs(alpha - 2.0) < 1e-9:
        # Official: a == 2 -> sqrt(a)*z == sqrt(2)*randn ~ N(0, 2),
        # which is the alpha=2 limit of the canonical S_alpha(0, 1) prior.
        return 2.0 * torch.ones(shape, device=device, dtype=torch.float32)
    a_raw = _sample_stable(
        alpha_inner=alpha / 2.0, beta=1.0, shape=shape, device=device, generator=generator
    )
    # Matches the official torchlevy is_isotropic branch:
    #     a = 2 * cos(pi * alpha / 4) ** (2 / alpha) * a_raw
    rescale = 2.0 * math.cos(math.pi * alpha / 4.0) ** (2.0 / alpha)
    a = rescale * a_raw
    if clamp_a is not None:
        a = a.clamp(min=0.0, max=clamp_a)
    else:
        a = a.clamp(min=0.0)
    return a


def sample_sas_from_a(a: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    r"""Symmetric alpha-stable noise via the augmented representation
    :math:`\epsilon = \sqrt{a}\, z`, with :math:`z \sim \mathcal{N}(0, I)`.
    """
    z = torch.randn(a.shape, device=a.device, dtype=a.dtype, generator=generator)
    return torch.sqrt(a.clamp(min=0.0)) * z


def sample_sas_noise(
    alpha: float,
    shape,
    device,
    generator: Optional[torch.Generator] = None,
    clamp_a: Optional[float] = 2000.0,
) -> Tensor:
    """Convenience: draw a fresh ``a`` and return ``sqrt(a) * randn``."""
    a = sample_skewed_levy(alpha, shape, device, generator=generator, clamp_a=clamp_a)
    return sample_sas_from_a(a, generator=generator)


# ----------------------------------------------------------------------------
# Cosine schedule, scale_preserving (matching dlpm.gen_noise_schedule)
# ----------------------------------------------------------------------------


class DLPMSchedule:
    r"""Cosine "scale-preserving" noise schedule from
    ``DLPM.gen_noise_schedule(scale='scale_preserving')``.

    Holds 1-D arrays (length ``num_steps``) for ``gammas, bar_gammas, sigmas,
    bar_sigmas``. With

    .. math::
        \bar{\gamma}_t^\alpha + \bar{\sigma}_t^\alpha = 1,

    the marginal :math:`x_t = \bar{\gamma}_t x_0 + \bar{\sigma}_t \epsilon`
    has unit alpha-norm at every step.
    """

    def __init__(self, alpha: float, num_steps: int, device: torch.device, s: float = 0.008):
        if not (0 < alpha <= 2):
            raise ValueError("alpha must lie in (0, 2]")
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = device

        timesteps = torch.arange(num_steps, dtype=torch.float32, device=device)
        schedule = torch.cos((timesteps / num_steps + s) / (1 + s) * math.pi / 2.0) ** 2
        bar_alpha_ddpm = schedule / schedule[0]
        betas = 1.0 - bar_alpha_ddpm / torch.cat([bar_alpha_ddpm[:1], bar_alpha_ddpm[:-1]])
        alphas = (1.0 - betas).clamp(min=1e-8)

        gammas = alphas.pow(1.0 / alpha)
        bar_gammas = torch.cumprod(gammas, dim=0)
        sigmas = (1.0 - gammas.pow(alpha)).clamp(min=0).pow(1.0 / alpha)
        bar_sigmas = (1.0 - bar_gammas.pow(alpha)).clamp(min=0).pow(1.0 / alpha)

        self.gammas = gammas
        self.bar_gammas = bar_gammas
        self.sigmas = sigmas
        self.bar_sigmas = bar_sigmas


# ----------------------------------------------------------------------------
# Training: get_one_rv_loss_elements, ported from dlpm.py
# ----------------------------------------------------------------------------


def dlpm_one_rv_loss_elements(
    schedule: DLPMSchedule,
    x_0: Tensor,
    t: Tensor,
) -> Tuple[Tensor, Tensor]:
    r"""Sample :math:`(x_t, \epsilon_t)` for the augmented one-r.v. DLPM
    training loss (faithful port of ``DLPM.get_one_rv_loss_elements``).

    For each sample we draw a positive :math:`\alpha/2`-stable :math:`a_t`,
    a Gaussian :math:`z_t`, and form

    .. math::
        x_t = \bar{\gamma}_t x_0 + \sqrt{a_t}\,\bar{\sigma}_t\, z_t,
        \qquad
        \epsilon_t = \sqrt{a_t}\, z_t.

    Marginalising over :math:`a_t` recovers the heavy-tailed S-alpha-S
    forward kernel; conditional on :math:`a_t`, both :math:`x_t` and
    :math:`\epsilon_t` are bounded for any individual sample, so the L2 loss
    against ``model(x_t, t/T)`` is well-defined per sample.

    Args:
        schedule: a :class:`DLPMSchedule`.
        x_0: ``[B, d]`` clean data.
        t: ``[B]`` integer timesteps in ``[1, num_steps - 1]``.

    Returns:
        ``(x_t, eps_t)`` both of shape ``[B, d]``.
    """
    a_t = sample_skewed_levy(schedule.alpha, x_0.shape, x_0.device)
    z_t = torch.randn_like(x_0)
    bg_t = schedule.bar_gammas[t].view(-1, *([1] * (x_0.dim() - 1)))
    bs_t = schedule.bar_sigmas[t].view(-1, *([1] * (x_0.dim() - 1)))
    sigma_prime_sqrt = torch.sqrt(a_t) * bs_t
    x_t = bg_t * x_0 + sigma_prime_sqrt * z_t
    eps_t = torch.sqrt(a_t) * z_t  # equals (x_t - bg_t * x_0) / bs_t
    return x_t, eps_t


def dlpm_eps_loss(eps_pred: Tensor, eps_t: Tensor) -> Tensor:
    r"""L2-norm epsilon-prediction loss, faithful port of
    ``GenerativeLevyProcess.compute_loss_terms`` with ``lploss == 2``:

    .. math::
        \mathcal{L} = \mathbb{E}_n \left[
            \sqrt{\tfrac{1}{d} \sum_i (\hat{\epsilon}_{n,i} - \epsilon_{n,i})^2}
        \right].

    The per-sample square root is critical for :math:`\alpha < 2`: the
    target :math:`\epsilon_t = \sqrt{a_t} z_t` has infinite second moment
    (because :math:`\mathbb{E}[a_t] = \infty`) so the squared loss
    :math:`\mathbb{E}[(\hat{\epsilon} - \epsilon_t)^2]` is infinite, but the
    L2-norm loss :math:`\mathbb{E}[\sqrt{a_t}\,|z_t|]` is finite as long as
    :math:`\mathbb{E}[\sqrt{a_t}] < \infty`, which holds for any
    :math:`\alpha > 1`.
    """
    per_sample_mse = (eps_pred - eps_t).pow(2).mean(dim=tuple(range(1, eps_pred.dim())))
    return per_sample_mse.clamp(min=1e-12).sqrt().mean()


# ----------------------------------------------------------------------------
# Sampling: anterior_mean_variance_dlim, ported from dlpm.py
# ----------------------------------------------------------------------------


@torch.no_grad()
def dlpm_p_sample(
    model: Callable[[Tensor, Tensor], Tensor],
    alpha: float,
    n_samples: int,
    n_steps: int,
    device: torch.device,
    d: int = 2,
    return_trajectory: bool = False,
    seed: Optional[int] = None,
    clamp_a: Optional[float] = 100.0,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Stochastic DLPM backward sampler, faithful port of
    ``DLPM.anterior_mean_variance_dlpm`` + ``p_sample_loop_progressive``.

    The full DLPM sampler is *augmented*: it precomputes a single chain of
    auxiliary positive Levy r.v.'s ``A[1:T]`` (one per particle, per step),
    builds the cumulative variances ``Sigmas[t]`` via the recurrence

    .. math::
        \Sigma_t = s_t^2 A_t + \gamma_t^2 \Sigma_{t-1},

    and then steps backwards using

    .. math::
        \Gamma_t &= 1 - \frac{\gamma_t^2 \Sigma_{t-1}}{\Sigma_t} \\
        x_{t-1} &= \frac{x_t - \bar{\sigma}_t \Gamma_t \hat{\epsilon}}{\gamma_t}
                  + \sqrt{\Gamma_t \Sigma_{t-1}}\, z

    with fresh Gaussian ``z`` per step. This is the sampler the official 2D
    config uses (``eval.dlpm.deterministic: false``).

    For NFE < num_train_steps we still use the full ``num_train_steps`` chain
    of auxiliary A's but only call the model at ``n_steps`` of the backward
    indices, sub-sampled uniformly. This matches the spirit of "rescale
    timesteps" in the official code.

    Args:
        model: callable ``(x [B, d], t [B]) -> eps_pred [B, d]``.
        schedule: a :class:`DLPMSchedule`.
        n_samples: batch size.
        n_steps: number of model evaluations (NFE).
        device: torch device.
        d: state dim.
        return_trajectory: if True, return ``[n_steps + 1, B, d]`` intermediates.
        seed: optional RNG seed.
        clamp_a: clamp the auxiliary positive Levy r.v.'s for stability;
            the official uses 2000, we default to 100 for the 2D toy where
            occasional huge a values otherwise destabilise the chain.

    Returns:
        ``(x_final, traj_or_None)``.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
    else:
        gen = None

    # Build a fresh schedule sized to the requested NFE. Cosine "scale-preserving"
    # schedule values only depend on fractional position t / T, so the model
    # (trained at T=1000) sees the same (x_t, t/T) joint at this rebuilt schedule
    # as at training time. This mirrors the official `rescale_diffusion`.
    schedule = DLPMSchedule(alpha=alpha, num_steps=n_steps, device=device)
    T = schedule.num_steps  # == n_steps
    g = schedule.gammas
    s = schedule.sigmas
    bs = schedule.bar_sigmas
    eps_clamp = 1e-8

    # 1) Sample the full A_{1:T} chain (one positive r.v. per (timestep, particle, dim))
    A = torch.stack([
        sample_skewed_levy(alpha, (n_samples, d), device,
                           generator=gen, clamp_a=clamp_a)
        for _ in range(T)
    ], dim=0)  # [T, B, d]

    # 2) Compute cumulative Sigmas[t] via the official recurrence
    Sigmas = [s[0] ** 2 * A[0]]
    for tt in range(1, T):
        Sigmas.append(s[tt] ** 2 * A[tt] + g[tt] ** 2 * Sigmas[-1])
    Sigmas = torch.stack(Sigmas, dim=0)  # [T, B, d]

    # 3) Initial noise: x_T = bs[T-1] * sqrt(a) * z, augmented form
    init_a = sample_skewed_levy(alpha, (n_samples, d), device,
                                generator=gen, clamp_a=clamp_a)
    init_z = torch.randn(n_samples, d, device=device, generator=gen)
    x_t = bs[-1] * torch.sqrt(init_a) * init_z
    traj = [x_t.detach().cpu().clone()] if return_trajectory else None

    # 4) Backward loop: iterate ALL T = n_steps indices (unit step), as the
    # official p_sample_loop does after rescale_diffusion.
    for i in range(T - 1, 0, -1):
        t_cur = i
        t_in = torch.full(
            (n_samples,), float(t_cur) / T, device=device, dtype=x_t.dtype
        )
        eps_pred = model(x_t, t_in)

        # anterior_mean_variance_dlpm at index t_cur
        Gamma_t = 1.0 - (g[t_cur] ** 2 * Sigmas[t_cur - 1]) / Sigmas[t_cur].clamp(min=eps_clamp)
        Gamma_t = Gamma_t.clamp(min=0.0)
        mean = (x_t - bs[t_cur] * Gamma_t * eps_pred) / g[t_cur].clamp(min=eps_clamp)
        var = Gamma_t * Sigmas[t_cur - 1]

        # No noise on the very last step (matches `nonzero_mask = (t != 1)`)
        nonzero = 1.0 if t_cur != 1 else 0.0
        z = torch.randn(n_samples, d, device=device, generator=gen)
        x_t = mean + nonzero * torch.sqrt(var.clamp(min=0.0)) * z

        if return_trajectory:
            traj.append(x_t.detach().cpu().clone())

    traj_tensor = torch.stack(traj, dim=0) if return_trajectory else None
    return x_t, traj_tensor


@torch.no_grad()
def dlpm_sample(
    model: Callable[[Tensor, Tensor], Tensor],
    alpha: float,
    n_samples: int,
    n_steps: int,
    device: torch.device,
    d: int = 2,
    return_trajectory: bool = False,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""DLIM-eta=0 deterministic backward sampler, faithful port of
    ``DLPM.anterior_mean_variance_dlim`` with ``eta = 0``:

    .. math::
        x_{t-1} = \frac{x_t - \bar{\sigma}_t \hat{\epsilon}}{\gamma_t}
                  + \bar{\sigma}_{t-1}\, \hat{\epsilon}.

    For :math:`\alpha = 2` this reduces to standard DDIM with :math:`\eta = 0`.

    Like the official ``GenerativeLevyProcess.sample`` with non-default
    ``reverse_steps``, we **rebuild a fresh schedule** sized to ``n_steps``
    and iterate all of its unit steps. This is well-defined because the
    cosine "scale-preserving" schedule values only depend on fractional
    position :math:`t / T`, so the model (trained at any T) sees the same
    :math:`(x_t, t/T)` joint at the rebuilt schedule as at training.

    The model is called with the rescaled time :math:`t / T \in [0, 1]`,
    matching ``GenerativeLevyProcess._scale_timesteps`` with
    ``rescale_timesteps=True`` (which is what the official 2D config uses).

    Args:
        model: callable ``(x [B, d], t [B]) -> eps_pred [B, d]``.
        alpha: stability index used during training.
        n_samples: batch size.
        n_steps: number of inference steps. NFE = ``n_steps``.
        device: torch device.
        d: state dim (default 2 for the toy notebook).
        return_trajectory: if True, also return ``[n_steps, B, d]`` of
            intermediate ``x_t`` (one per unit step).
        seed: optional RNG seed.

    Returns:
        ``(x_final, traj_or_None)``.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
    else:
        gen = None

    schedule = DLPMSchedule(alpha=alpha, num_steps=n_steps, device=device)
    T = schedule.num_steps  # == n_steps
    g = schedule.gammas
    bs = schedule.bar_sigmas
    eps_clamp = 1e-8

    # Initial noise: x_T ~ bs[T-1] * S-alpha-S, drawn via the augmented form.
    init_noise = sample_sas_noise(alpha, (n_samples, d), device, generator=gen)
    x_t = bs[-1] * init_noise
    traj = [x_t.detach().cpu().clone()] if return_trajectory else None

    # Iterate unit steps from T-1 down to 1, as the official p_sample_loop does.
    for t_cur in range(T - 1, 0, -1):
        t_in = torch.full(
            (n_samples,), float(t_cur) / T, device=device, dtype=x_t.dtype
        )
        eps_pred = model(x_t, t_in)

        # anterior_mean_variance_dlim with eta=0:
        # sample = (x_t - bs[t]*eps) / g[t] + bs[t-1]*eps
        x_t = (x_t - bs[t_cur] * eps_pred) / g[t_cur].clamp(min=eps_clamp) \
              + bs[t_cur - 1] * eps_pred

        if return_trajectory:
            traj.append(x_t.detach().cpu().clone())

    traj_tensor = torch.stack(traj, dim=0) if return_trajectory else None
    return x_t, traj_tensor
