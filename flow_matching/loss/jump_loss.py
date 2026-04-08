# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Jump model loss for continuous spaces (R^d).
# Based on Generator Matching (Holderrieth et al., ICLR 2025),
# Appendix D.2 (ELBO loss) and Appendix F (Euclidean jump model details).

import torch
import torch.nn as nn
from torch import Tensor


def compute_bin_centers(num_bins: int, device: torch.device) -> Tensor:
    """Compute evenly spaced bin centers in [-1, 1].

    Args:
        num_bins: number of bins.
        device: torch device.

    Returns:
        Tensor of shape [num_bins] with bin center values.
    """
    # Bins evenly spaced in [-1, 1], centers at midpoints
    edges = torch.linspace(-1.0, 1.0, num_bins + 1, device=device)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers


def compute_conditional_jump_kernel(
    x_t: Tensor,
    x_1: Tensor,
    t: Tensor,
    bin_centers: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the conditional jump rate kernel Q_t^z(bin_j; x_t | z=x_1).

    From paper eq. (13) and Appendix E.7:
        k_t(x) = x^2 - (t+1)*x*z - (1-t)^2 + t*z^2
        lambda_t(x|z) = [k_t(x)]_+ / (1-t)^3
        J_t(x'|z) proportional to [-k_t(x')]_+ * p_t(x'|z)
        Q_t(x'; x|z) = lambda_t(x|z) * J_t(x'|z)

    Per-dimension, independent (Proposition 4).

    Args:
        x_t: current state, shape [B, C, H, W], values in [-1, 1].
        x_1: target data, shape [B, C, H, W], values in [-1, 1].
        t: time, shape [B].
        bin_centers: bin center values, shape [num_bins].

    Returns:
        Q_z: conditional jump rate kernel, shape [B, C, H, W, num_bins].
        lambda_z: conditional jump intensity, shape [B, C, H, W].
    """
    # Expand t to match spatial dims: [B] -> [B, 1, 1, 1]
    t_exp = t[:, None, None, None]
    one_minus_t = 1.0 - t_exp

    # k_t(x_t) for current state, per dimension
    # eq. (13): k_t(x) = x^2 - (t+1)*x*z - (1-t)^2 + t*z^2
    k_t_xt = (
        x_t ** 2
        - (t_exp + 1.0) * x_t * x_1
        - one_minus_t ** 2
        + t_exp * x_1 ** 2
    )

    # Jump intensity: lambda_t(x|z) = [k_t(x)]_+ / (1-t)^3
    # Clamp (1-t) to avoid division by zero near t=1
    one_minus_t_clamped = one_minus_t.clamp(min=1e-5)
    lambda_z = torch.relu(k_t_xt) / (one_minus_t_clamped ** 3)

    # Jump distribution J_t over bins
    # For each bin center c_j, compute:
    #   [-k_t(c_j)]_+ * N(c_j; t*z, (1-t)^2)
    # bin_centers: [num_bins] -> [1, 1, 1, 1, num_bins]
    bc = bin_centers[None, None, None, None, :]

    # x_1 expanded: [B, C, H, W] -> [B, C, H, W, 1]
    z = x_1.unsqueeze(-1)
    t_bc = t_exp.unsqueeze(-1)
    one_minus_t_bc = one_minus_t.unsqueeze(-1)

    # k_t at bin centers
    k_t_bins = (
        bc ** 2
        - (t_bc + 1.0) * bc * z
        - one_minus_t_bc ** 2
        + t_bc * z ** 2
    )

    # [-k_t(c_j)]_+ : regions where jump distribution has mass
    neg_k_positive = torch.relu(-k_t_bins)

    # Gaussian weight: N(c_j; t*z, (1-t)^2)
    mean = t_bc * z
    var = one_minus_t_bc ** 2
    var_clamped = var.clamp(min=1e-8)
    log_gauss = -0.5 * (bc - mean) ** 2 / var_clamped
    gauss = torch.exp(log_gauss)  # unnormalized is fine, we normalize J

    # J_t proportional to [-k_t(c_j)]_+ * gaussian
    J_unnorm = neg_k_positive * gauss  # [B, C, H, W, num_bins]
    J_sum = J_unnorm.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    J_z = J_unnorm / J_sum  # normalized jump distribution

    # Q_z = lambda_z * J_z
    Q_z = lambda_z.unsqueeze(-1) * J_z  # [B, C, H, W, num_bins]

    return Q_z, lambda_z


class JumpLoss(nn.Module):
    """ELBO/KL loss for the jump model on continuous spaces.

    Based on paper eq. (270) and Appendix F eq. (362-365):
        L = E[ sum_{x' != x_t} Q_theta(x'; x_t) - Q_z(x'; x_t) * log Q_theta(x'; x_t) ]

    This is a Bregman divergence (the continuous-time ELBO).

    Args:
        num_bins: number of bins for discretizing [-1, 1].
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, num_bins: int = 256, reduction: str = "mean"):
        super().__init__()
        self.num_bins = num_bins
        self.reduction = reduction
        # Register bin centers as buffer (not a parameter)
        self.register_buffer(
            "bin_centers", compute_bin_centers(num_bins, torch.device("cpu"))
        )

    def forward(
        self,
        jump_logits: Tensor,
        jump_intensity_logit: Tensor,
        x_t: Tensor,
        x_1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute the jump ELBO loss.

        Args:
            jump_logits: model output logits for jump distribution,
                shape [B, C, num_bins, H, W].
            jump_intensity_logit: model output for jump intensity (pre-softplus),
                shape [B, C, H, W].
            x_t: current interpolated state, shape [B, C, H, W], values in [-1, 1].
            x_1: target data, shape [B, C, H, W], values in [-1, 1].
            t: time, shape [B].

        Returns:
            Scalar loss (or per-element if reduction='none').
        """
        bin_centers = self.bin_centers.to(x_t.device)

        # Model predictions
        # lambda_theta >= 0 via softplus
        lambda_theta = torch.nn.functional.softplus(jump_intensity_logit)
        # J_theta: categorical distribution over bins
        J_theta = torch.softmax(jump_logits, dim=2)  # [B, C, num_bins, H, W]
        # Q_theta = lambda_theta * J_theta
        Q_theta = lambda_theta.unsqueeze(2) * J_theta  # [B, C, num_bins, H, W]

        # Compute conditional Q_z from analytic formula
        # Q_z shape: [B, C, H, W, num_bins]
        Q_z, _ = compute_conditional_jump_kernel(x_t, x_1, t, bin_centers)
        # Permute to [B, C, num_bins, H, W] to match Q_theta
        Q_z = Q_z.permute(0, 1, 4, 2, 3)

        # Find the bin that x_t falls into (for excluding self-transitions)
        # x_t is in [-1, 1], map to bin index
        # bin_width = 2.0 / num_bins
        bin_indices = ((x_t + 1.0) / 2.0 * self.num_bins).long()
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)

        # Create mask: 1 for all bins except the current bin (x' != x_t)
        # bin_indices: [B, C, H, W] -> [B, C, 1, H, W]
        mask = torch.ones_like(Q_theta)
        mask.scatter_(2, bin_indices.unsqueeze(2), 0.0)

        # ELBO loss: sum_{x' != x_t} [ Q_theta(x') - Q_z(x') * log(Q_theta(x')) ]
        # From eq. (270)
        log_Q_theta = torch.log(Q_theta.clamp(min=1e-10))
        loss = mask * (Q_theta - Q_z * log_Q_theta)

        # Sum over bins dimension
        loss = loss.sum(dim=2)  # [B, C, H, W]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
