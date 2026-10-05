"""Independent role encoders with bidirectional cross-attention fusion."""

from __future__ import annotations

import torch
from torch import nn


def make_encoder(d_model: int, n_heads: int, layers: int,
                 dim_feedforward: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward,
                                       dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model),
                                 enable_nested_tensor=False)


class DualEncoder(nn.Module):
    """Both musical streams contribute to fused decoder memory.

    Encoders have independent parameters; token embeddings remain shared. Padding
    is masked both in self-attention and in the cross-stream fusion attention.
    """

    def __init__(self, d_model: int, n_heads: int, layers: int,
                 dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.solo_encoder = make_encoder(d_model, n_heads, layers, dim_feedforward, dropout)
        self.acc_encoder = make_encoder(d_model, n_heads, layers, dim_feedforward, dropout)
        self.solo_to_acc = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.acc_to_solo = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.solo_norm = nn.LayerNorm(d_model)
        self.acc_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, solo: torch.Tensor, acc: torch.Tensor,
                solo_mask: torch.Tensor, acc_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        solo = self.solo_encoder(solo, src_key_padding_mask=solo_mask)
        acc = self.acc_encoder(acc, src_key_padding_mask=acc_mask)
        solo_cross, _ = self.solo_to_acc(solo, acc, acc, key_padding_mask=acc_mask,
                                        need_weights=False)
        acc_cross, _ = self.acc_to_solo(acc, solo, solo, key_padding_mask=solo_mask,
                                      need_weights=False)
        memory = torch.cat((self.solo_norm(solo + self.dropout(solo_cross)),
                            self.acc_norm(acc + self.dropout(acc_cross))), dim=1)
        return memory, torch.cat((solo_mask, acc_mask), dim=1)

