# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Gaussian Mixture Flow Matching (GMFlow).
# Reference: Chen, Zhang, Tan, Xu, Luan, Guibas, Wetzstein, Bi.
# "Gaussian Mixture Flow Matching Models." ICML 2025.
# Code follows the official implementation at https://github.com/Lakonik/GMFlow

from .loss import gm_nll_loss, gm_transition_nll_loss
from .ops import (
    gm_to_iso_gaussian,
    gm_to_mean,
    gm_to_sample,
    iso_gaussian_mul_iso_gaussian,
    gm_mul_iso_gaussian,
    u_to_x0_gm,
    reverse_transition_gm,
    denoising_gm_convert_to_mean,
)
from .solver import GMFlowSolver

__all__ = [
    "gm_nll_loss",
    "gm_transition_nll_loss",
    "gm_to_iso_gaussian",
    "gm_to_mean",
    "gm_to_sample",
    "iso_gaussian_mul_iso_gaussian",
    "gm_mul_iso_gaussian",
    "u_to_x0_gm",
    "reverse_transition_gm",
    "denoising_gm_convert_to_mean",
    "GMFlowSolver",
]
