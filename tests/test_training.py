"""Meaningful learning, differentiability, leakage, and exact resume checks."""

from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from src.models.seq2seq import ModelConfig, Seq2SeqTransformer  # noqa: E402
from src.training.collator import TranscriptionCollator, model_inputs  # noqa: E402
from src.training.dataset import TranscriptionDataset, Vocabulary, load_prepared_dataset  # noqa: E402
from src.training.losses import melody_presence_loss, transcription_loss  # noqa: E402
from src.training.trainer import TrainingConfig, load_checkpoint, train_model  # noqa: E402


def event(pitch, role="PIANO", position=0):
    return [f"<POS_{position}>", f"<ROLE_{role}>", f"<PITCH_{pitch}>",
            "<DUR_24>", "<VELOCITY_BIN_9>", "<VOICE_ID_0>"]


def segment(index, split="train", work=None):
    return {"segment_id": f"segment_{index}", "work_id": work or f"work_{index}", "split": split,
            "source_solo_tokens": ["<BAR_1>", *event(60 + index, "SOLO")],
            "source_acc_tokens": ["<BAR_1>", *event(48 + index, "ACC")],
            "target_piano_tokens": ["<BAR_1>", *event(60 + index), *event(48 + index)],
            "technique_features": {"melody_coverage_octave_tolerant": 1.0},
            "controls": {"melody_preservation": 1.0, "octave_doubling": 0.0}}


def write_dataset(path, segments):
    path.write_text(json.dumps({"schema_version": 1, "subdivisions": 24,
                               "segments": segments}), encoding="utf-8")
    return path


def test_complete_vocab_contains_unseen_pitches_and_roundtrips():
    vocab = Vocabulary.from_segments([segment(0)])
    tokens = ["<BAR_8>", "<POS_47>", "<ROLE_PIANO>", "<PITCH_127>", "<DUR_384>", "<VOICE_ID_12>"]
    assert vocab.decode(vocab.encode(tokens, add_bos=True, add_eos=True)) == tokens
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.token_to_id == vocab.token_to_id
    with pytest.raises(ValueError, match="outside"):
        vocab.encode(["<PITCH_128>"])


def test_teacher_forcing_shift_padding_and_explicit_controls():
    first, second = segment(0), segment(1)
    second["target_piano_tokens"] += event(73, position=24)
    second["controls"]["octave_doubling"] = 1.0
    vocab = Vocabulary.from_segments([first, second])
    data = TranscriptionDataset([first, second], vocab)
    batch = TranscriptionCollator(vocab)([data[0], data[1]])
    assert batch["target_input_ids"][0, 0] == vocab.bos_id
    assert batch["labels"][0, 0] == vocab.token_to_id["<BAR_1>"]
    assert batch["labels"][0, len(data[0]["target_ids"]) - 2] == vocab.eos_id
    assert batch["target_padding_mask"][0, -1]
    assert batch["labels"][0, -1] == vocab.pad_id
    assert batch["controls"][0, 1] == 0 and batch["controls"][1, 1] == 1
    assert batch["melody_positions"].shape == (2, 1, 2)
    assert batch["labels"][0, batch["melody_positions"][0, 0, 0]] == vocab.token_to_id["<PITCH_60>"]


def test_work_split_leakage_fails_before_training(tmp_path):
    path = write_dataset(tmp_path / "data.json", [segment(0, work="same"), segment(1, "test", work="same")])
    with pytest.raises(ValueError, match="multiple splits"):
        load_prepared_dataset(path)


def test_long_sequences_are_not_silently_truncated():
    vocab = Vocabulary.from_segments([segment(0)])
    with pytest.raises(ValueError, match="resegment"):
        TranscriptionDataset([segment(0)], vocab, max_sequence_length=10)


def test_instrumental_intro_without_solo_trains_dual_streams(tmp_path):
    intro = segment(0)
    intro["source_solo_tokens"] = []
    intro["target_piano_tokens"] = ["<BAR_1>", *event(48)]
    path = write_dataset(tmp_path / "intro.json", [intro])
    prepared = load_prepared_dataset(path)
    vocab = Vocabulary.from_segments(prepared.segments)
    data = TranscriptionDataset(prepared.segments, vocab)
    assert data[0]["solo_ids"] == [vocab.bos_id, vocab.eos_id]
    batch = TranscriptionCollator(vocab)([data[0]])
    assert batch["melody_positions"].shape[1] == 0
    model = Seq2SeqTransformer(len(vocab), ModelConfig(architecture="dual", d_model=16, n_heads=2,
                                                       encoder_layers=1, decoder_layers=1,
                                                       dim_feedforward=32, dropout=0.0,
                                                       max_sequence_length=128))
    logits = model(**model_inputs(batch))
    values = transcription_loss(logits, batch["labels"], pad_id=vocab.pad_id,
                                melody_positions=batch["melody_positions"], melody_pitch_ids=batch["melody_pitch_ids"],
                                lambda_melody=0.1)
    assert torch.isfinite(values["loss"]) and values["melody_loss"] == 0
    values["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.acc_encoder.parameters())


def test_soft_melody_loss_is_differentiable_and_rewards_allowed_pitch():
    logits = torch.zeros(1, 2, 8, requires_grad=True)
    positions = torch.tensor([[[0, 1]]])
    pitches = torch.tensor([[[3, 4, -1]]])
    loss = melody_presence_loss(logits, positions, pitches)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, :, 3].sum() < 0  # Increasing allowed pitch decreases loss.
    improved = logits.detach().clone()
    improved[:, :, 3] = 3
    assert melody_presence_loss(improved, positions, pitches) < loss
    ignored = melody_presence_loss(logits, torch.full((1, 1, 1), -1), pitches)
    assert ignored == 0


def test_cross_entropy_ignores_padding():
    logits = torch.randn(1, 3, 10)
    labels = torch.tensor([[4, 5, 0]])
    first = transcription_loss(logits, labels, pad_id=0)["loss"]
    logits[:, 2] = 100 * torch.randn(10)
    second = transcription_loss(logits, labels, pad_id=0)["loss"]
    torch.testing.assert_close(first, second)


def test_tiny_training_learns_and_checkpoint_reproduces_logits(tmp_path):
    path = write_dataset(tmp_path / "data.json", [segment(0), segment(1), segment(2, "validation")])
    model_config = ModelConfig(d_model=32, n_heads=2, encoder_layers=1, decoder_layers=1,
                               dim_feedforward=64, dropout=0.0, max_sequence_length=128)
    config = TrainingConfig(epochs=25, batch_size=2, learning_rate=0.02, weight_decay=0.0,
                             seed=7, num_threads=1, save_every=25)
    report = train_model(path, tmp_path / "training", model_config=model_config,
                         training_config=config, progress=False)
    assert report["final_token_loss"] < report["initial_token_loss"] * 0.25
    assert report["final_token_accuracy"] > 0.8
    loaded, vocab, metadata = load_checkpoint(report["checkpoint"])
    again, _, _ = load_checkpoint(report["checkpoint"])
    batch = TranscriptionCollator(vocab)([TranscriptionDataset([segment(0)], vocab)[0]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(**model_inputs(batch)), again(**model_inputs(batch)))
    assert metadata["train_work_ids"] == ["work_0", "work_1"]
    assert metadata["validation_work_ids"] == ["work_2"]
    assert metadata["subdivisions"] == 24


def test_resume_matches_uninterrupted_dropout_training(tmp_path):
    path = write_dataset(tmp_path / "data.json", [segment(0), segment(1)])
    model_config = ModelConfig(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1,
                               dim_feedforward=32, dropout=0.1, max_sequence_length=128)
    config = TrainingConfig(epochs=4, batch_size=1, learning_rate=0.01, weight_decay=0.0,
                             seed=9, num_threads=1, save_every=2)
    complete = train_model(path, tmp_path / "complete", model_config=model_config,
                           training_config=config, progress=False)
    half = train_model(path, tmp_path / "resume", model_config=model_config,
                       training_config=replace(config, epochs=2), progress=False)
    resumed = train_model(path, tmp_path / "resume", model_config=model_config,
                          training_config=config, resume=half["checkpoint"], progress=False)
    left, _, _ = load_checkpoint(complete["checkpoint"])
    right, _, _ = load_checkpoint(resumed["checkpoint"])
    assert resumed["global_step"] == complete["global_step"] == 8
    for name, parameter in left.state_dict().items():
        torch.testing.assert_close(parameter, right.state_dict()[name], rtol=0, atol=0)


def test_incomplete_and_corrupt_checkpoints_fail_explicitly(tmp_path):
    incomplete = tmp_path / "incomplete.pt"
    torch.save({"type": "virtuosotranscribe.transformer", "schema_version": 1}, incomplete)
    with pytest.raises(ValueError, match="Incomplete"):
        load_checkpoint(incomplete)
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"this is not a torch checkpoint")
    with pytest.raises(ValueError, match="Unreadable or invalid"):
        load_checkpoint(corrupt)


