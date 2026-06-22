"""Token-stream fuser (ported from seamstress TokenStreamFuser).

Fuses N per-modality streams, each (B, T, Di), into one token stream (B, T, output_dim):
  per stream:  LayerNorm(Di) -> Linear(Di -> pre_fuse)
  then:        concat -> LayerNorm -> Linear -> GELU -> Linear(-> output_dim)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class TokenStreamFuser(nn.Module):
    def __init__(self, input_dims: Sequence[int], output_dim: int, pre_fuse_dim: int, post_fuse_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_dims = tuple(int(d) for d in input_dims)
        self.stream_norms = nn.ModuleList([nn.LayerNorm(d) for d in self.input_dims])
        self.stream_projs = nn.ModuleList([nn.Linear(d, pre_fuse_dim) for d in self.input_dims])
        fused_in = len(self.input_dims) * pre_fuse_dim
        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, post_fuse_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(post_fuse_dim, output_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, streams: Sequence[Tensor]) -> Tensor:
        assert len(streams) == len(self.input_dims), "stream count mismatch"
        projected = [self.stream_projs[i](self.stream_norms[i](s)) for i, s in enumerate(streams)]
        return self.fuse(torch.cat(projected, dim=-1))
