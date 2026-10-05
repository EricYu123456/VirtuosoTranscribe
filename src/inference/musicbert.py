"""Optional real MusicBERT embedding adapter for content reranking.

Load an upstream fairseq checkpoint and dictionary, or a TorchScript export of
its feature extractor. No pretrained checkpoint is bundled or fabricated.
Octuple bins follow microsoft/muzic/musicbert/preprocess.py (MIT license).
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.representation.events import ScoreData


_TIME_SIGNATURES = [(numerator, 2 ** power) for power in range(7)
                    for numerator in range(1, 2 ** power * 2 + 1)]
_DURATION_BINS = [index * 16 + offset for index in range(8)
                  for offset in range(16) for _ in range(2 ** index)]


def octuple_tokens(score: ScoreData) -> list[str]:
    """Upstream OctupleMIDI fields on a quantized 1/16-quarter grid.

    Content comparison normalizes every instrument to piano, and uses declared
    measure/pickup positions. This adapter intentionally reports unsupported
    signatures and oversized segments instead of silently dropping notes.
    """
    measures = {measure.number: measure for measure in score.measures}
    if not score.notes:
        raise ValueError("MusicBERT requires non-empty note events")
    if len(measures) > 256:
        raise ValueError("MusicBERT content scoring requires segments of at most 256 measures")
    bar_indices = {number: index for index, number in enumerate(sorted(measures))}
    result = ["<s>"] * 8
    for note in sorted(score.notes, key=lambda n: (n.measure, n.onset, n.pitch, n.duration)):
        if note.measure not in measures:
            raise ValueError("MusicBERT note references missing measure metadata")
        measure = measures[note.measure]
        numerator, denominator = map(int, measure.time_signature.split("/"))
        if (numerator, denominator) not in _TIME_SIGNATURES:
            raise ValueError(f"Unsupported MusicBERT time signature {measure.time_signature}")
        position = round((note.onset - measure.onset + measure.padding_left) * 16)
        if not 0 <= position < 128:
            raise ValueError("MusicBERT position is outside the upstream 0..127 dictionary")
        duration = max(0, round(note.duration * 16))
        duration_bin = _DURATION_BINS[min(duration, len(_DURATION_BINS) - 1)]
        bpm = next((tempo.bpm for tempo in reversed(sorted(score.tempos, key=lambda t: t.onset))
                    if tempo.onset <= note.onset), 120.)
        tempo_bin = round(math.log2(max(16., min(256., bpm)) / 16.) * 12)
        fields = (bar_indices[note.measure], position, 0, note.pitch, duration_bin,
                  note.velocity // 4, _TIME_SIGNATURES.index((numerator, denominator)), tempo_bin)
        result.extend(f"<{field}-{value}>" for field, value in enumerate(fields))
    result.extend(["</s>"] * 8)
    return result


class MusicBERTContentScorer:
    """Cosine similarity from an explicitly supplied pretrained encoder.

    TorchScript contract: forward(input_ids[B,S]) -> features[B,S,D] or [B,D].
    dictionary_path is the exact fairseq dict.txt used to train the encoder.
    For upstream fairseq, user_dir points to muzic/musicbert/musicbert and
    dictionary_path may be its containing binarized data directory.
    """

    def __init__(self, checkpoint: str | Path, dictionary_path: str | Path, *,
                 backend: str = "torchscript", user_dir: str | Path | None = None,
                 device: str = "cpu", max_tokens: int = 8192):
        import torch
        self.device = device
        self.max_tokens = max_tokens
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise ValueError(f"MusicBERT checkpoint does not exist: {checkpoint}")
        dictionary_path = Path(dictionary_path)
        if backend == "torchscript":
            if dictionary_path.is_dir():
                dictionary_path /= "dict.txt"
            symbols = ["<s>", "<pad>", "</s>", "<unk>"]
            for line in dictionary_path.read_text(encoding="utf-8").splitlines():
                token = line.rsplit(" ", 1)[0]
                if token not in symbols:
                    symbols.append(token)
            self.dictionary = {token: index for index, token in enumerate(symbols)}
            self.model = torch.jit.load(str(checkpoint), map_location=device).eval()
            self.backend = backend
        elif backend == "fairseq":
            try:
                from fairseq import utils
                from fairseq.models.roberta import RobertaModel
            except ImportError as exc:
                raise ValueError("Upstream MusicBERT needs a compatible fairseq installation; use a TorchScript export on unsupported platforms") from exc
            if user_dir is None:
                raise ValueError("Upstream MusicBERT requires --musicbert-user-dir for its registered model")
            utils.import_user_module(SimpleNamespace(user_dir=str(Path(user_dir).resolve())))
            self.model = RobertaModel.from_pretrained(
                str(checkpoint.parent), checkpoint_file=checkpoint.name,
                data_name_or_path=str(dictionary_path if dictionary_path.is_dir() else dictionary_path.parent))
            self.model.to(device).eval()
            self.dictionary = self.model.task.source_dictionary.indices
            self.backend = backend
        else:
            raise ValueError("MusicBERT backend must be torchscript or fairseq")
        self._source_cache: dict[tuple[str, ...], Any] = {}

    def embed(self, score: ScoreData):
        import torch
        tokens = octuple_tokens(score)
        if len(tokens) > self.max_tokens:
            raise ValueError(f"MusicBERT input has {len(tokens)} tokens; segment it to the configured {self.max_tokens} limit")
        absent = [token for token in tokens if token not in self.dictionary]
        if absent:
            raise ValueError(f"MusicBERT dictionary is incompatible with octuple encoding: {absent[:3]}")
        ids = torch.tensor([[self.dictionary[token] for token in tokens]],
                           dtype=torch.long, device=self.device)
        with torch.no_grad():
            features = (self.model.extract_features(ids) if self.backend == "fairseq"
                        else self.model(ids))
        if not isinstance(features, torch.Tensor) or features.ndim not in (2, 3):
            raise ValueError("MusicBERT feature extractor must return [B,D] or [B,S,D] tensor")
        vector = features.mean(1) if features.ndim == 3 else features
        if not torch.isfinite(vector).all() or vector.norm() == 0:
            raise ValueError("MusicBERT returned invalid or zero embeddings")
        return torch.nn.functional.normalize(vector, dim=-1)

    def __call__(self, source: ScoreData, target: ScoreData) -> float:
        source_key = tuple(octuple_tokens(source))
        if source_key not in self._source_cache:
            self._source_cache[source_key] = self.embed(source)
        cosine = float((self._source_cache[source_key] * self.embed(target)).sum())
        return max(0., min(1., (cosine + 1) / 2))
