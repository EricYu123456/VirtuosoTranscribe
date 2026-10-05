from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.inference.neural_generate import decode_with_model, load_neural_controls, tokens_to_piano_score  # noqa: E402
from src.models.constraints import GenerationError  # noqa: E402
from src.representation.events import MeasureInfo, NoteEvent, PartInfo, ScoreData, SourceScore, TempoEvent  # noqa: E402
from src.training.dataset import Vocabulary  # noqa: E402


def make_source():
    solo = (NoteEvent(72, 0, 1, "solo", role="SOLO"),
            NoteEvent(74, 1, 1, "solo", role="SOLO"))
    acc = (NoteEvent(48, 0, 4, "acc", role="ACC"),)
    score = ScoreData(solo + acc, (PartInfo("solo", "Violin", "Violin"),
                                  PartInfo("acc", "Piano", "Piano")),
                      (MeasureInfo(1, 0, 4),), tempos=(TempoEvent(0, 92),))
    return SourceScore(score, solo, acc, ("solo",), ("acc",))


def musical_event(position, pitch, duration=12):
    return [f"<POS_{position}>", "<ROLE_PIANO>", f"<PITCH_{pitch}>", f"<DUR_{duration}>",
            "<VELOCITY_BIN_9>", "<VOICE_ID_0>"]


class ScriptedDecoder(torch.nn.Module):
    """Predict a known nonmelodic sequence to distinguish constraint modes."""

    def __init__(self, vocab, tokens=None, conditioned=False):
        super().__init__()
        self.vocab = vocab
        self.preferred = vocab.encode(tokens or ["<BAR_1>", *musical_event(0, 55),
                                                 *musical_event(24, 57), "<EOS>"])
        self.config = SimpleNamespace(max_sequence_length=256, conditioned=conditioned)
        self.grad_modes = []
        self.encode_calls = 0
        self.controls = None

    def encode(self, source_ids, **kwargs):
        self.encode_calls += 1
        self.controls = kwargs["controls"]
        self.grad_modes.append(torch.is_grad_enabled())
        return source_ids

    def decode(self, prefix_ids, encoded):
        self.grad_modes.append(torch.is_grad_enabled())
        logits = torch.full((1, prefix_ids.shape[1], len(self.vocab)), -20.)
        index = min(prefix_ids.shape[1] - 1, len(self.preferred) - 1)
        logits[0, -1, self.preferred[index]] = 20.
        return logits


def test_none_soft_and_hard_are_distinct_real_decoding_modes():
    vocab = Vocabulary()
    source = make_source()
    plain = decode_with_model(ScriptedDecoder(vocab), vocab, source,
                              melody_mode="none", temperature=0)[0]
    soft = decode_with_model(ScriptedDecoder(vocab), vocab, source,
                             melody_mode="soft", temperature=0, soft_strength=2)[0]
    hard = decode_with_model(ScriptedDecoder(vocab), vocab, source,
                             melody_mode="hard", temperature=0)[0]
    assert [n.pitch for n in plain.score.notes] == [55, 57]
    assert [n.pitch for n in soft.score.notes] == [55, 57]
    for source_note in source.solo:
        assert any(note.pitch in (source_note.pitch, source_note.pitch - 12, source_note.pitch + 12)
                   and note.onset == source_note.onset and note.duration == source_note.duration
                   for note in hard.score.notes)
    assert hard.report["hard_emitted_melody_events"] == 2
    assert hard.report["projection_repairs"] == 0
    assert hard.report["constraint_forced_token_steps"] == 4
    assert plain.report["constraint_forced_token_steps"] == soft.report["constraint_forced_token_steps"] == 0


def test_eval_no_grad_cached_source_and_continuous_conditioning():
    vocab = Vocabulary()
    model = ScriptedDecoder(vocab, conditioned=True)
    model.train()
    controls = {"figuration_density": .8, "harmonic_freedom": .2, "melody_preservation": .9}
    candidates = decode_with_model(model, vocab, make_source(), controls=controls,
                                   num_candidates=2, seed=7, melody_mode="none", temperature=0)
    assert not model.training
    assert not any(model.grad_modes)
    assert model.encode_calls == 1
    assert model.controls.shape == (1, 13)
    assert float(model.controls[0, 6]) == pytest.approx(.8)
    assert candidates[0].report["controls_conditioned"]
    assert candidates[0].score == candidates[1].score


def test_bounded_failure_does_not_return_partial_or_baseline_notes():
    vocab = Vocabulary()
    with pytest.raises(GenerationError, match="without EOS"):
        decode_with_model(ScriptedDecoder(vocab), vocab, make_source(),
                          melody_mode="none", temperature=0, max_new_tokens=8)


def test_beam_generation_is_valid_and_keeps_complete_metadata():
    vocab = Vocabulary()
    result = decode_with_model(ScriptedDecoder(vocab), vocab, make_source(),
                               melody_mode="hard", temperature=0, beam_width=2)[0]
    assert result.score.measures == make_source().score.measures
    assert result.score.tempos == make_source().score.tempos
    assert result.report["decoding"] == "beam"
    assert result.report["hard_emitted_melody_events"] == 2


def test_nucleus_sampling_has_reproducible_seeds():
    vocab = Vocabulary()
    first = decode_with_model(ScriptedDecoder(vocab), vocab, make_source(),
                              melody_mode="none", seed=4, num_candidates=2)
    second = decode_with_model(ScriptedDecoder(vocab), vocab, make_source(),
                               melody_mode="none", seed=4, num_candidates=2)
    assert [(r.tokens, r.report) for r in first] == [(r.tokens, r.report) for r in second]


def test_beam_nbest_uses_distinct_completed_paths():
    vocab = Vocabulary()
    candidates = decode_with_model(ScriptedDecoder(vocab), vocab, make_source(),
                                   melody_mode="hard", beam_width=3, num_candidates=3)
    assert len(candidates) == 3
    assert len({tuple(candidate.tokens) for candidate in candidates}) == 3
    assert [candidate.report["beam_rank"] for candidate in candidates] == [0, 1, 2]


def test_malformed_model_output_and_score_tokens_are_explicit():
    vocab = Vocabulary()
    class InvalidDecoder(ScriptedDecoder):
        def decode(self, prefix_ids, encoded):
            return super().decode(prefix_ids, encoded).fill_(float("nan"))
    with pytest.raises(GenerationError, match="non-finite"):
        decode_with_model(InvalidDecoder(vocab), vocab, make_source())
    with pytest.raises(ValueError, match="Incomplete"):
        tokens_to_piano_score(make_source(), ["<BAR_1>", "<POS_0>"])
    class WrongShape(ScriptedDecoder):
        def decode(self, prefix_ids, encoded):
            return torch.zeros(1, len(vocab))
    with pytest.raises(GenerationError, match="logits with shape"):
        decode_with_model(WrongShape(vocab), vocab, make_source())


def test_neural_controls_accept_named_freedoms_and_reject_invalid_values(tmp_path):
    preset = tmp_path / "controls.yaml"
    preset.write_text("figuration_density: 0.7\nharmonic_freedom: 0.2\nmelody_preservation: 0.95\n", encoding="utf-8")
    controls = load_neural_controls(preset)
    assert controls["figuration_density"] == .7
    assert controls["melody_preservation"] == .95
    with pytest.raises(ValueError, match="Unknown"):
        load_neural_controls({"imaginary_operator": 1})
    with pytest.raises(ValueError, match="finite"):
        load_neural_controls({"figuration_density": float("nan")})


def test_optional_musicbert_runs_supplied_feature_model_and_dictionary(tmp_path):
    from src.inference.musicbert import MusicBERTContentScorer, octuple_tokens
    class FeatureEncoder(torch.nn.Module):
        def forward(self, ids):
            values = ids.float()
            return torch.stack((values.mean(1), values.std(1) + 1), dim=-1)
    score = make_source().score
    tokens = list(dict.fromkeys(octuple_tokens(score)))
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("\n".join(f"{token} 1" for token in tokens if token not in {"<s>", "</s>"}), encoding="utf-8")
    checkpoint = tmp_path / "feature_contract_smoke_test.pt"
    torch.jit.trace(FeatureEncoder(), torch.ones(1, 16, dtype=torch.long)).save(str(checkpoint))
    scorer = MusicBERTContentScorer(checkpoint, dictionary)
    assert scorer(score, score) == pytest.approx(1.)
    assert len(scorer._source_cache) == 1


def test_reranking_reports_harmony_techniques_and_optional_content():
    from src.baselines.rule_based import arrange_rule_based
    from src.inference.rerank import rank_candidates
    source = make_source()
    candidate = arrange_rule_based(source)
    calls = []
    def scorer(left, right):
        calls.append((left, right))
        return .8
    ranking = rank_candidates(source, [candidate], {"harmonic_freedom": .2}, content_scorer=scorer)
    assert calls
    assert ranking[0]["score_components"]["musicbert_content_similarity"] == .8
    assert "harmonic_consistency" in ranking[0]["score_components"]
    assert "figuration_density" in ranking[0]["technique_observations"]


def test_missing_neural_checkpoint_never_falls_back_to_baseline(tmp_path, capsys):
    from src.inference.export import export_score
    from src.inference.generate import main
    input_path = export_score(make_source().score, tmp_path / "input.musicxml")
    with pytest.raises(SystemExit) as error:
        main(["--input", str(input_path), "--checkpoint", str(tmp_path / "missing.pt"),
              "--output-dir", str(tmp_path / "output"), "--solo-part", "Violin"])
    assert error.value.code == 2
    assert not (tmp_path / "output" / "best.musicxml").exists()
    assert "Traceback" not in capsys.readouterr().err


def test_corrupt_checkpoint_has_clean_generation_error(tmp_path):
    from src.inference.neural_generate import generate_neural_results
    checkpoint = tmp_path / "invalid.pt"
    checkpoint.write_bytes(b"invalid checkpoint bytes")
    with pytest.raises(GenerationError, match="checkpoint"):
        generate_neural_results(make_source(), checkpoint)
