# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# GM-SDE / GM-ODE solvers for GMFlow, ported from
#     https://github.com/Lakonik/GMFlow/blob/main/lib/models/diffusions/gmflow.py
#     https://github.com/Lakonik/GMFlow/blob/main/lib/models/diffusions/schedulers/gmflow_sde.py
#
# We follow the rectified-flow noise schedule (alpha_t = 1 - t, sigma_t = t)
# with t in [0, 1]. The model output is a velocity-space GM with
#   means      [B, K, D]
#   logstds    [B, 1]
#   logweights [B, K]   (already log_softmax'd)

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from .ops import (
    denoising_gm_convert_to_mean,
    gm_mul_iso_gaussian,
    gm_to_iso_gaussian,
    gm_to_mean,
    gm_to_sample,
    iso_gaussian_mul_iso_gaussian,
    u_to_x0_gm,
)


class GMFlowSolver:
    """Backward sampler for GMFlow.

    Supports four configurations matching the paper's Figure 2 / Algorithm 2:
        mode="sde", order=1   ->  GM-SDE  (1st order)
        mode="sde", order=2   ->  GM-SDE  2  (2nd order, Adams-Bashforth-style)
        mode="ode", order=1   ->  GM-ODE  (1st order, with sub-steps)
        mode="ode", order=2   ->  GM-ODE  2 (2nd order, with sub-steps)

    Args:
        model: callable ``(x [B, D], t [B]) -> dict(means [B, K, D],
            logstds [B, 1], logweights [B, K])`` returning a velocity-space GM.
        mode: ``"sde"`` (GM-SDE, sample mode) or ``"ode"`` (GM-ODE, mean mode).
        order: 1 or 2.
        gm2_ca, gm2_cb: 2nd-order rescaling constants from the official code
            (defaults match `gm2_coefs=[0.005, 1.0]`).
    """

    def __init__(
        self,
        model: Callable[[Tensor, Tensor], dict],
        mode: str = "sde",
        order: int = 1,
        gm2_ca: float = 0.005,
        gm2_cb: float = 1.0,
    ):
        if mode not in ("sde", "ode"):
            raise ValueError("mode must be 'sde' or 'ode'")
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.model = model
        self.mode = mode
        self.order = order
        self.gm2_ca = gm2_ca
        self.gm2_cb = gm2_cb

    @staticmethod
    def _build_sigmas(num_steps: int, device: torch.device) -> Tensor:
        """Linear sigma schedule from sigma_max=1 down to 0.

        Matches the official scheduler:
            sigmas = 1 - np.linspace(0, 1, num_steps, endpoint=False)
            sigmas = cat([sigmas, [0]])
        which yields ``num_steps + 1`` values, with the last being 0 (so we
        get exactly ``num_steps`` non-degenerate steps).
        """
        base = torch.arange(num_steps, dtype=torch.float32, device=device) / num_steps
        sigmas = 1.0 - base                                  # length num_steps
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)], 0)  # length num_steps + 1
        return sigmas

    @staticmethod
    def _step_coefs(sigma_high: float, sigma_low: float, eps: float = 1e-6):
        """Compute (c1, c2, c3, c3_sqrt) for the SDE step
            x_{t-Δt} = c1 * x_t + c2 * x_0 + c3_sqrt * noise
        """
        sigma_h = max(sigma_high, eps)
        alpha_h = 1.0 - sigma_high
        alpha_l = 1.0 - sigma_low
        sl_over_sh = sigma_low / sigma_h
        ah_over_al = alpha_h / max(alpha_l, eps)
        beta_over_sh_sq = 1.0 - (sl_over_sh * ah_over_al) ** 2
        c1 = sl_over_sh ** 2 * ah_over_al
        c2 = beta_over_sh_sq * alpha_l
        c3 = beta_over_sh_sq * sigma_low ** 2
        c3 = max(c3, 0.0)
        return c1, c2, c3, c3 ** 0.5

    @torch.no_grad()
    def _gm2_correct(
        self,
        gm_x0: dict,
        gauss: dict,
        x_t: Tensor,
        sigma: float,
        h: float,
        prev_state: Optional[dict],
    ) -> Tuple[dict, dict, dict]:
        """Adams-Bashforth-style 2nd-order correction (paper App. A.2).

        Updates the *base* GM by an isotropic Gaussian factor whose mean is
        the rescaled difference ``mean_now - mean_from_prev`` (extrapolated to
        the next mid-point). Returns the updated GM and Gaussian outputs as
        well as the new prev_state.
        """
        new_state = {"gm": gm_x0, "x_t": x_t, "sigma": sigma, "h": h}
        if prev_state is None:
            return gm_x0, gauss, new_state

        gm_mean_now = gauss["mean"]                       # [B, D]
        avg_var = gauss["var"]                            # [B, 1]

        prev_gm = prev_state["gm"]
        prev_x_t = prev_state["x_t"]
        prev_sigma = prev_state["sigma"]
        prev_h = prev_state["h"]

        mean_from_prev = denoising_gm_convert_to_mean(
            prev_gm, x_t, prev_x_t, sigma, prev_sigma
        )                                                  # [B, D]

        k = 0.5 * h / max(prev_h, 1e-6)
        err_power = avg_var * (self.gm2_cb ** 2 + self.gm2_ca)
        scale = (1.0 - err_power / (prev_h ** 2 + 1e-12)).clamp(min=0.0).sqrt() * k
        bias = (gm_mean_now - mean_from_prev) * scale     # [B, D]

        bias_power = bias.square().mean(dim=-1, keepdim=True)  # [B, 1]
        bias = bias * (avg_var / bias_power.clamp(min=1e-6)).clamp(max=1.0).sqrt()
        bias_power = bias.square().mean(dim=-1, keepdim=True)

        new_gauss = {
            "mean": gauss["mean"] + bias,
            "var": gauss["var"] * (1.0 - bias_power / avg_var.clamp(min=1e-6)).clamp(min=1e-6),
        }
        # Reweight the GM by g_new / g_old to absorb the bias.
        ratio = iso_gaussian_mul_iso_gaussian(new_gauss, gauss, p1=1.0, p2=-1.0)
        gm_x0 = gm_mul_iso_gaussian(gm_x0, ratio, gm_power=1.0, gauss_power=1.0)

        return gm_x0, new_gauss, new_state

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_steps: int,
        device: torch.device,
        d: int = 2,
        n_substeps: Optional[int] = None,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the backward sampler.

        Args:
            n_samples: batch size.
            n_steps: number of NFE = number of network calls.
            device: torch device.
            d: state dim (default 2 for the toy notebook).
            n_substeps: only used for ``mode="ode"``. ``None`` -> 1 substep
                (pure 1st-order ODE step). Following the paper, we default to
                ``ceil(128 / n_steps)`` for n_steps < 16, else 2.
            return_trajectory: if True, return ``[n_steps + 1, B, D]`` of
                intermediate ``x_t``.
            seed: optional RNG seed for reproducible noise.

        Returns:
            ``(x_final, traj_or_None)``.
        """
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
        else:
            gen = None

        if self.mode == "ode" and n_substeps is None:
            n_substeps = max(1, (128 + n_steps - 1) // n_steps) if n_steps < 16 else 2
        if self.mode == "sde":
            n_substeps = 1

        # Initial noise x_T ~ N(0, I) (sigma = 1).
        x_t = torch.randn(n_samples, d, device=device, generator=gen)
        traj = [x_t.clone().cpu()] if return_trajectory else None

        sigmas = self._build_sigmas(n_steps, device)  # [n_steps + 1]
        prev_state: Optional[dict] = None

        for i in range(n_steps):
            sigma = float(sigmas[i].item())
            sigma_next = float(sigmas[i + 1].item())
            h = sigma - sigma_next

            t_in = torch.full((n_samples,), sigma, device=device, dtype=x_t.dtype)
            gm_u = self.model(x_t, t_in)
            gm_x0 = u_to_x0_gm(gm_u, x_t, sigma)
            gauss = gm_to_iso_gaussian(gm_x0)

            if self.order == 2:
                gm_x0, gauss, prev_state = self._gm2_correct(
                    gm_x0, gauss, x_t, sigma, h, prev_state
                )

            # Sub-step loop. For SDE, n_substeps == 1.
            #
            # In both modes we follow the official GMFlowSDEScheduler.step
            # formula  x_{t-Δt} = c1*x_t + c2*x_0 + c3_sqrt*noise. The only
            # difference between SDE and ODE is the choice of x_0:
            #   - SDE: x_0 ~ q_θ(x_0 | x_t)               (sample)
            #   - ODE: x_0 = E[q_θ(x_0 | x_t)]            (mean)
            # Both are stochastic. The "ODE" name follows the official code
            # (which uses a single GMFlowSDEScheduler with `output_mode='mean'`)
            # and refers to using a deterministic prediction rather than a
            # deterministic transition.
            x_t_base = x_t
            sigma_base = sigma
            for sub in range(n_substeps):
                sigma_curr = sigma + (sigma_next - sigma) * sub / n_substeps
                sigma_target = (
                    sigma_next if (sub == n_substeps - 1) else
                    sigma + (sigma_next - sigma) * (sub + 1) / n_substeps
                )
                if sub == 0:
                    if self.mode == "sde":
                        x0 = gm_to_sample(gm_x0, generator=gen)
                    else:  # ode
                        x0 = gauss["mean"]
                else:
                    # Refine x_0 estimate at the new (x_t, sigma_curr) without
                    # querying the network, via eq. (10).
                    x0 = denoising_gm_convert_to_mean(
                        gm_x0, x_t, x_t_base, sigma_curr, sigma_base
                    )

                c1, c2, _, c3_sqrt = self._step_coefs(sigma_curr, sigma_target)
                noise = torch.randn(
                    n_samples, d, device=device, dtype=x_t.dtype, generator=gen
                )
                x_t = c1 * x_t + c2 * x0 + c3_sqrt * noise

            if return_trajectory:
                traj.append(x_t.clone().cpu())

        traj_tensor = torch.stack(traj, dim=0) if return_trajectory else None
        return x_t, traj_tensor
