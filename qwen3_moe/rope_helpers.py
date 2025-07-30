# SPDX-FileCopyrightText: © 2023
# SPDX-License-Identifier: MIT

from typing import Tuple, Any

import torch


def precompute_freqs_cis(config: Any):
    theta = config.rope_theta
    dim = config.head_dim

    indices = torch.arange(start=0, end=dim, step=2, dtype=torch.float32, requires_grad=False)[: (dim // 2)] / dim
    freqs = torch.reciprocal(torch.pow(theta, indices)).to(dtype=torch.float32)
    t = torch.arange(start=0, end=config.max_seq_len, step=1, dtype=torch.float32, requires_grad=False)  # type: ignore
    freqs = torch.outer(t, freqs).to(dtype=torch.float32)  # type: ignore
    freqs_cis = torch.polar(abs=torch.ones_like(input=freqs, dtype=torch.float32), angle=freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.to(dtype=torch.float16), xk_out.to(dtype=torch.float16)

