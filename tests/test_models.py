"""Functional attention tests for actual learned model architectures."""

import pytest

torch = pytest.importorskip("torch")

from src.models.seq2seq import ModelConfig, Seq2SeqTransformer  # noqa: E402
from src.models.conditioning import controls_to_vector  # noqa: E402


@pytest.fixture(autouse=True)
def deterministic_model_tests():
    torch.manual_seed(4)
    torch.set_num_threads(1)


def tiny(**changes):
    return ModelConfig(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1,
                       dim_feedforward=32, dropout=0.0, max_sequence_length=32, **changes)


def test_decoder_attention_is_causal():
    model = Seq2SeqTransformer(32, tiny()).eval()
    source = torch.tensor([[1, 7, 8, 2]])
    left = torch.tensor([[1, 9, 10, 11]])
    right = torch.tensor([[1, 9, 22, 23]])
    with torch.no_grad():
        a, b = model(source, left), model(source, right)
    torch.testing.assert_close(a[:, :2], b[:, :2], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(a[:, 2:], b[:, 2:])


def test_source_padding_does_not_change_valid_decoder_predictions():
    model = Seq2SeqTransformer(32, tiny()).eval()
    target = torch.tensor([[1, 9, 10]])
    with torch.no_grad():
        short = model(torch.tensor([[1, 7, 8, 2]]), target)
        padded = model(torch.tensor([[1, 7, 8, 2, 0, 0]]), target)
    torch.testing.assert_close(short, padded, rtol=1e-5, atol=1e-6)


def test_encoder_memory_can_be_cached_for_generation():
    model = Seq2SeqTransformer(32, tiny()).eval()
    source, target = torch.tensor([[1, 7, 2]]), torch.tensor([[1, 8]])
    with torch.no_grad():
        cached = model.decode(target, model.encode(source))
        direct = model(source, target)
    torch.testing.assert_close(cached, direct)


def test_dual_streams_receive_gradients_and_affect_predictions():
    model = Seq2SeqTransformer(32, tiny(architecture="dual"))
    source = torch.tensor([[1, 7, 8, 2]])
    solo = torch.tensor([[1, 7, 2]])
    acc = torch.tensor([[1, 8, 2]])
    target = torch.tensor([[1, 9, 10]])
    logits = model(source, target, solo_ids=solo, acc_ids=acc)
    logits.square().mean().backward()
    for encoder in (model.encoder.solo_encoder, model.encoder.acc_encoder,
                    model.encoder.solo_to_acc, model.encoder.acc_to_solo):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in encoder.parameters())
    with torch.no_grad():
        changed = model(source, target, solo_ids=solo, acc_ids=torch.tensor([[1, 15, 2]]))
    assert not torch.allclose(logits, changed)


def test_continuous_conditioning_changes_output_and_has_gradient():
    model = Seq2SeqTransformer(32, tiny(conditioned=True))
    source, target = torch.tensor([[1, 7, 2]]), torch.tensor([[1, 8]])
    faithful = torch.tensor([controls_to_vector({"octave_doubling": 0.0})])
    doubled = torch.tensor([controls_to_vector({"octave_doubling": 1.0})])
    first, second = model(source, target, controls=faithful), model(source, target, controls=doubled)
    assert not torch.allclose(first, second)
    second.square().mean().backward()
    assert model.conditioning.projection[0].weight.grad.abs().sum() > 0


def test_empty_accompaniment_uses_bos_eos_and_all_padding_rejected():
    model = Seq2SeqTransformer(32, tiny(architecture="dual"))
    source, target = torch.tensor([[1, 7, 2]]), torch.tensor([[1, 8]])
    logits = model(source, target, solo_ids=source, acc_ids=torch.tensor([[1, 2]]))
    assert torch.isfinite(logits).all()
    with pytest.raises(ValueError, match="entirely of padding"):
        model(source, target, solo_ids=source, acc_ids=torch.tensor([[0, 0]]))


def test_sequence_limits_and_invalid_controls_are_explicit():
    model = Seq2SeqTransformer(32, tiny(conditioned=True))
    source, target = torch.ones(1, 33, dtype=torch.long), torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="exceeds model limit"):
        model(source, target)
    with pytest.raises(ValueError, match="within 0..1"):
        model(torch.tensor([[1, 7, 2]]), target, controls=torch.full((1, 13), 2.0))
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=15, n_heads=2)

