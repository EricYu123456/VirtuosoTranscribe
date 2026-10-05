"""Causal encoder-decoder Transformer with optional role separation and controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn

from .conditioning import CONTROL_NAMES, TechniqueConditioning
from .dual_encoder import DualEncoder, make_encoder


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "single"
    d_model: int = 128
    n_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_sequence_length: int = 4096
    conditioned: bool = False
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.architecture not in {"single", "dual"}:
            raise ValueError("architecture must be 'single' or 'dual'")
        for name in ("d_model", "n_heads", "encoder_layers", "decoder_layers",
                     "dim_feedforward", "max_sequence_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in 0..1")
        if not isinstance(self.conditioned, bool):
            raise ValueError("conditioned must be a boolean")
        if isinstance(self.pad_id, bool) or not isinstance(self.pad_id, int) or self.pad_id < 0:
            raise ValueError("pad_id must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelConfig:
        payload = dict(payload)
        aliases = {"encoder_decoder_transformer": "single",
                   "dual_encoder_transformer": "dual"}
        payload["architecture"] = aliases.get(payload.get("architecture"), payload.get("architecture", "single"))
        return cls(**payload)


@dataclass
class EncodedSource:
    memory: torch.Tensor
    padding_mask: torch.Tensor
    conditioning: torch.Tensor | None = None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, dropout: float) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                            * (-math.log(10000) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"Token sequence length {embeddings.shape[1]} exceeds model limit {self.encoding.shape[1]}; segment the score instead of truncating it")
        return self.dropout(embeddings + self.encoding[:, :embeddings.shape[1]].to(embeddings.dtype))


class Seq2SeqTransformer(nn.Module):
    """Shared interface for the vanilla, dual, and conditioned ablations.

    ``encode`` caches source memory for autoregressive inference. ``decode``
    always uses a causal attention mask, including while teacher forcing.
    """

    def __init__(self, vocab_size: int, config: ModelConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = ModelConfig.from_dict(config) if isinstance(config, dict) else config
        config = self.config
        if vocab_size <= config.pad_id:
            raise ValueError("Vocabulary must contain the configured padding token")
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=config.pad_id)
        self.positions = PositionalEncoding(config.d_model, config.max_sequence_length, config.dropout)
        self.role_embedding = nn.Embedding(2, config.d_model)
        if config.architecture == "dual":
            self.encoder = DualEncoder(config.d_model, config.n_heads, config.encoder_layers,
                                       config.dim_feedforward, config.dropout)
        else:
            self.encoder = make_encoder(config.d_model, config.n_heads, config.encoder_layers,
                                        config.dim_feedforward, config.dropout)
        layer = nn.TransformerDecoderLayer(config.d_model, config.n_heads, config.dim_feedforward,
                                           config.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, config.decoder_layers, norm=nn.LayerNorm(config.d_model))
        self.output = nn.Linear(config.d_model, vocab_size)
        self.conditioning = TechniqueConditioning(config.d_model) if config.conditioned else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # PyTorch cloned layers otherwise begin with identical initialization.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.embedding.weight[self.config.pad_id].zero_()

    def _mask(self, ids: torch.Tensor, supplied: torch.Tensor | None) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("Token IDs must have non-empty shape [batch, sequence]")
        mask = ids.eq(self.config.pad_id) if supplied is None else supplied.to(device=ids.device, dtype=torch.bool)
        if mask.shape != ids.shape:
            raise ValueError("Padding mask shape must equal token ID shape")
        if mask.all(dim=1).any():
            raise ValueError("A token stream cannot consist entirely of padding; use BOS/EOS for an empty accompaniment")
        return mask

    def _embed(self, ids: torch.Tensor, role: int | None = None) -> torch.Tensor:
        embeddings = self.embedding(ids) * math.sqrt(self.config.d_model)
        if role is not None:
            embeddings = embeddings + self.role_embedding.weight[role]
        return self.positions(embeddings)

    def encode(self, source_ids: torch.Tensor, *, source_padding_mask: torch.Tensor | None = None,
               controls: torch.Tensor | None = None, solo_ids: torch.Tensor | None = None,
               acc_ids: torch.Tensor | None = None, solo_padding_mask: torch.Tensor | None = None,
               acc_padding_mask: torch.Tensor | None = None) -> EncodedSource:
        if self.config.architecture == "dual":
            if solo_ids is None or acc_ids is None:
                raise ValueError("Dual encoding requires both solo_ids and acc_ids")
            solo_mask = self._mask(solo_ids, solo_padding_mask)
            acc_mask = self._mask(acc_ids, acc_padding_mask)
            memory, mask = self.encoder(self._embed(solo_ids, 0), self._embed(acc_ids, 1),
                                        solo_mask, acc_mask)
        else:
            mask = self._mask(source_ids, source_padding_mask)
            memory = self.encoder(self._embed(source_ids), src_key_padding_mask=mask)
        condition = None
        if self.conditioning is not None:
            if controls is None:
                controls = memory.new_zeros((memory.shape[0], len(CONTROL_NAMES)))
                controls[:, 0] = 1.0
            controls = controls.to(device=memory.device, dtype=memory.dtype)
            if controls.shape[0] != memory.shape[0]:
                raise ValueError("Technique controls batch size must equal source batch size")
            condition = self.conditioning(controls)
            memory = torch.cat((condition.unsqueeze(1), memory), dim=1)
            mask = torch.cat((mask.new_zeros((mask.shape[0], 1)), mask), dim=1)
        return EncodedSource(memory, mask, condition)

    def decode(self, prefix_ids: torch.Tensor, encoded_source: EncodedSource, *,
               target_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        padding = self._mask(prefix_ids, target_padding_mask)
        embeddings = self._embed(prefix_ids)
        if encoded_source.conditioning is not None:
            embeddings = embeddings + encoded_source.conditioning.unsqueeze(1)
        length = prefix_ids.shape[1]
        causal_mask = torch.ones((length, length), device=prefix_ids.device, dtype=torch.bool).triu(1)
        hidden = self.decoder(embeddings, encoded_source.memory, tgt_mask=causal_mask,
                              tgt_key_padding_mask=padding,
                              memory_key_padding_mask=encoded_source.padding_mask)
        return self.output(hidden)

    def forward(self, source_ids: torch.Tensor, target_input_ids: torch.Tensor, *,
                source_padding_mask: torch.Tensor | None = None,
                target_padding_mask: torch.Tensor | None = None,
                controls: torch.Tensor | None = None, solo_ids: torch.Tensor | None = None,
                acc_ids: torch.Tensor | None = None, solo_padding_mask: torch.Tensor | None = None,
                acc_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.encode(source_ids, source_padding_mask=source_padding_mask, controls=controls,
                             solo_ids=solo_ids, acc_ids=acc_ids,
                             solo_padding_mask=solo_padding_mask, acc_padding_mask=acc_padding_mask)
        return self.decode(target_input_ids, memory, target_padding_mask=target_padding_mask)

