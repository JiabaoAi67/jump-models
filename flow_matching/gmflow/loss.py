# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# GMFlow training losses, ported from
#     https://github.com/Lakonik/GMFlow/blob/main/lib/models/losses/diffusion_loss.py
#
# Two variants:
#   * gm_nll_loss            -- vanilla velocity NLL (paper eq. 6)
#   * gm_transition_nll_loss -- transition-loss variant (paper Sec. 3.4),
#                               equivalent to running the predicted velocity GM
#                               through the analytic reverse transition first
#                               and scoring the slightly-noised x_t_low.

import torch
from torch import Tensor

from .ops import reverse_transition_gm


def _gm_nll(means: Tensor, logstds: Tensor, logweights: Tensor, target: Tensor, eps: float = 1e-4) -> Tensor:
    """Negative log-likelihood of ``target`` under a GM with ``means [B,K,D]``,
    one shared scalar ``logstds [B,1]`` and ``logweights [B,K]``.

    Returns ``[B]``.
    """
    inv_std = (-logstds).exp().clamp(max=1.0 / eps).unsqueeze(-1)        # [B, 1, 1]
    diff = (means - target.unsqueeze(-2)) * inv_std                       # [B, K, D]
    log_normal = (-0.5 * diff.square() - logstds.unsqueeze(-1)).sum(-1)   # [B, K]
    return -torch.logsumexp(log_normal + logweights, dim=-1)              # [B]


def gm_nll_loss(means: Tensor, logstds: Tensor, logweights: Tensor, target: Tensor) -> Tensor:
    """Velocity-space GM NLL loss (paper eq. (6)).

    Args:
        means:      ``[B, K, D]`` predicted GM means in velocity space.
        logstds:    ``[B, 1]`` shared log-std of the GM.
        logweights: ``[B, K]`` log-softmaxed mixture weights.
        target:     ``[B, D]`` ground-truth velocity ``u = (x_t - x_0) / sigma``.

    Returns:
        Scalar loss (mean over batch).
    """
    return _gm_nll(means, logstds, logweights, target).mean()


def gm_transition_nll_loss(
    pred_gm_u: dict,
    x_t_high: Tensor,
    x_t_low: Tensor,
    sigma_high: float,
    sigma_low: float,
) -> Tensor:
    """Transition-loss variant (paper Sec. 3.4).

    Instead of scoring the velocity directly, push the predicted velocity GM
    through the analytic reverse transition (eq. (9)) to obtain a GM over
    ``x_t_low`` and score the *true* ``x_t_low`` under it. The transition GM
    has a lower-bounded variance ``c_3 + (c_2 σ_x)^2`` so the loss stays
    well-behaved when the predicted ``s`` becomes small.

    Args:
        pred_gm_u: dict with ``means [B,K,D]``, ``logstds [B,1]``, ``logweights [B,K]``,
                   the model's velocity-space prediction at ``x_t_high``.
        x_t_high:  ``[B, D]`` noisy data at the higher noise level.
        x_t_low:   ``[B, D]`` slightly less noisy data at the lower noise level.
        sigma_high, sigma_low: the corresponding noise levels in ``[0, 1]``,
                               with ``sigma_low < sigma_high``.

    Returns:
        Scalar loss (mean over batch).
    """
    trans_gm = reverse_transition_gm(pred_gm_u, x_t_high, sigma_high, sigma_low)
    return _gm_nll(trans_gm["means"], trans_gm["logstds"], trans_gm["logweights"], x_t_low).mean()
