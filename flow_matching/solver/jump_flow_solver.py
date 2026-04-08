# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Jump + Flow Markov Superposition solver for continuous spaces (R^d).
# Based on Generator Matching (Holderrieth et al., ICLR 2025),
# Algorithm 2 and Appendix F.

from math import ceil
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from flow_matching.solver.solver import Solver
from flow_matching.loss.jump_loss import compute_bin_centers
from flow_matching.utils import ModelWrapper


def jump_survival_probability(
    lambda_t: Tensor, t: float, h: float
) -> Tensor:
    """Compute P[No jump in [t, t+h)] using the corrected schedule R_{t,t+h}.

    From paper Appendix F, eq. (349)-(356):
    The jump intensity lambda_t(x) = [k_t(x)]_+ / (1-t)^3 has a (1-t)^3
    denominator that blows up near t=1. Instead of naive Euler
    P[no jump] = exp(-h * lambda), we integrate exactly over the (1-t-s)^3
    denominator while keeping [k_t(x)]_+ approximately constant:

        P[No Jump] = exp( lambda_t * (1-t) / 2 * (1 - (1-t)^2 / (1-t-h)^2) )

    This made a big difference in practice (FID 12 -> 4.5 on CIFAR-10).

    Args:
        lambda_t: jump intensity at time t, shape [...].
        t: current time (scalar).
        h: step size (scalar).

    Returns:
        P[no jump] with same shape as lambda_t.
    """
    one_minus_t = 1.0 - t
    one_minus_t_h = 1.0 - t - h

    if one_minus_t_h <= 1e-6:
        # At the very end, force jump with probability 1
        return torch.zeros_like(lambda_t)

    ratio_sq = (one_minus_t / one_minus_t_h) ** 2
    exponent = lambda_t * one_minus_t / 2.0 * (1.0 - ratio_sq)

    return torch.exp(exponent)


class JumpFlowEulerSolver(Solver):
    """Euler sampler for jump + flow Markov superposition on R^d.

    Implements Algorithm 2 from the paper. At each time step, for each
    spatial dimension independently:
    1. Compute flow velocity u_t and jump kernel (lambda_t, J_t).
    2. Flow step: x_flow = x_t + h * u_t
    3. Jump step: sample target bin from J_t, get x_jump = bin_center
    4. Decide per-dimension: if jump occurs, use x_jump; otherwise x_flow.

    The jump probability uses the corrected R_{t,t+h} schedule (eq. 356)
    instead of naive 1 - exp(-h * lambda).

    Args:
        model: trained model that outputs (velocity, jump_logits, jump_intensity).
        num_bins: number of bins for the jump distribution.
    """

    def __init__(
        self,
        model: ModelWrapper,
        num_bins: int = 256,
    ):
        super().__init__()
        self.model = model
        self.num_bins = num_bins
        self.register_buffer(
            "bin_centers",
            compute_bin_centers(num_bins, torch.device("cpu")),
        )

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float = 0.01,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        **model_extras,
    ) -> Tensor:
        """Sample from the jump+flow Markov superposition.

        Args:
            x_init: initial noise, shape [B, C, H, W].
            step_size: time step size.
            time_grid: time interval [t_start, t_end].
            return_intermediates: whether to return all intermediate states.
            **model_extras: extra args to model (e.g., label, cfg_scale).

        Returns:
            Final samples, shape [B, C, H, W].
        """
        device = x_init.device
        bin_centers = self.bin_centers.to(device)

        time_grid = time_grid.to(device)
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()

        n_steps = ceil((t_final - t_init) / step_size)
        t_discretization = torch.linspace(
            t_init, t_final, n_steps + 1, device=device
        )

        x_t = x_init.clone()
        intermediates = [x_t.clone()] if return_intermediates else []

        for i in range(n_steps):
            t_val = t_discretization[i].item()
            h_val = t_discretization[i + 1].item() - t_val
            t_tensor = t_discretization[i : i + 1]

            # Forward pass: model returns dict with velocity, jump_logits, jump_intensity
            model_out = self.model(
                x=x_t,
                t=t_tensor.repeat(x_t.shape[0]),
                **model_extras,
            )

            velocity = model_out["velocity"]  # [B, C, H, W]
            jump_logits = model_out["jump_logits"]  # [B, C, num_bins, H, W]
            jump_intensity_logit = model_out["jump_intensity"]  # [B, C, H, W]

            # Jump intensity (non-negative)
            lambda_t = F.softplus(jump_intensity_logit)

            # --- Flow step (Algorithm 2, line 5, no diffusion sigma=0) ---
            x_flow = x_t + h_val * velocity

            # --- Jump step (Algorithm 2, lines 2-4) ---
            # Sample jump target from J_theta
            J_theta = torch.softmax(jump_logits, dim=2)  # [B, C, num_bins, H, W]

            # Sample categorical per dimension
            B, C, nb, H, W = J_theta.shape
            # Reshape for sampling: [B*C*H*W, num_bins]
            J_flat = J_theta.permute(0, 1, 3, 4, 2).reshape(-1, nb)
            bin_idx_flat = torch.multinomial(J_flat.clamp(min=1e-8), 1).squeeze(-1)
            bin_idx = bin_idx_flat.reshape(B, C, H, W)

            # Map bin indices to actual values
            x_jump = bin_centers[bin_idx]  # [B, C, H, W]

            # --- Jump probability using R_{t,t+h} (eq. 356) ---
            p_no_jump = jump_survival_probability(lambda_t, t_val, h_val)
            mask_jump = torch.rand_like(p_no_jump) > p_no_jump

            # --- Combine (Algorithm 2, line 6) ---
            # Per-dimension: jump overrides flow
            x_t = torch.where(mask_jump, x_jump, x_flow)

            # Clamp to valid range
            x_t = x_t.clamp(-1.0, 1.0)

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t


class JumpOnlyEulerSolver(Solver):
    """Euler sampler for pure jump model (no flow component).

    Same as JumpFlowEulerSolver but without the flow step.
    When no jump occurs at a dimension, the value stays unchanged.

    Args:
        model: trained model that outputs jump_logits and jump_intensity.
        num_bins: number of bins.
    """

    def __init__(
        self,
        model: ModelWrapper,
        num_bins: int = 256,
    ):
        super().__init__()
        self.model = model
        self.num_bins = num_bins
        self.register_buffer(
            "bin_centers",
            compute_bin_centers(num_bins, torch.device("cpu")),
        )

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float = 0.01,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        **model_extras,
    ) -> Tensor:
        """Sample from the pure jump model.

        Args:
            x_init: initial noise, shape [B, C, H, W].
            step_size: time step size.
            time_grid: time interval.
            return_intermediates: whether to return all intermediate states.
            **model_extras: extra args to model.

        Returns:
            Final samples, shape [B, C, H, W].
        """
        device = x_init.device
        bin_centers = self.bin_centers.to(device)

        time_grid = time_grid.to(device)
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()

        n_steps = ceil((t_final - t_init) / step_size)
        t_discretization = torch.linspace(
            t_init, t_final, n_steps + 1, device=device
        )

        x_t = x_init.clone()
        intermediates = [x_t.clone()] if return_intermediates else []

        for i in range(n_steps):
            t_val = t_discretization[i].item()
            h_val = t_discretization[i + 1].item() - t_val
            t_tensor = t_discretization[i : i + 1]

            model_out = self.model(
                x=x_t,
                t=t_tensor.repeat(x_t.shape[0]),
                **model_extras,
            )

            jump_logits = model_out["jump_logits"]  # [B, C, num_bins, H, W]
            jump_intensity_logit = model_out["jump_intensity"]  # [B, C, H, W]

            lambda_t = F.softplus(jump_intensity_logit)

            # Sample jump target
            J_theta = torch.softmax(jump_logits, dim=2)
            B, C, nb, H, W = J_theta.shape
            J_flat = J_theta.permute(0, 1, 3, 4, 2).reshape(-1, nb)
            bin_idx_flat = torch.multinomial(J_flat.clamp(min=1e-8), 1).squeeze(-1)
            bin_idx = bin_idx_flat.reshape(B, C, H, W)
            x_jump = bin_centers[bin_idx]

            # Jump probability using R_{t,t+h}
            p_no_jump = jump_survival_probability(lambda_t, t_val, h_val)
            mask_jump = torch.rand_like(p_no_jump) > p_no_jump

            # No flow: if no jump, keep current value
            x_t = torch.where(mask_jump, x_jump, x_t)
            x_t = x_t.clamp(-1.0, 1.0)

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t
