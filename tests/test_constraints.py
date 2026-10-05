from dataclasses import replace

import pytest

from src.models.constraints import GenerationError, GrammarState, PianoTokenGrammar
from src.representation.events import MeasureInfo, NoteEvent, PartInfo, ScoreData, SourceScore


def source_score(solo=None):
    solo = tuple(solo or [NoteEvent(72, 0, 1, "solo", role="SOLO"),
                          NoteEvent(74, 1, 1, "solo", role="SOLO")])
    acc = (NoteEvent(48, 0, 4, "acc", role="ACC"),)
    score = ScoreData(solo + acc, (PartInfo("solo", "Violin"), PartInfo("acc", "Piano")),
                      (MeasureInfo(1, 0, 4),), title="Constraint fixture")
    return SourceScore(score, solo, acc, ("solo",), ("acc",))


def vocabulary():
    # Constraints themselves can be tested without importing optional PyTorch.
    values = ["<EOS>", "<ROLE_PIANO>"]
    for field, numbers in (("BAR", range(1, 3)), ("POS", range(96)),
                           ("PITCH", range(128)), ("DUR", range(1, 97)),
                           ("VELOCITY_BIN", range(16)), ("VOICE_ID", range(4))):
        values.extend(f"<{field}_{i}>" for i in numbers)
    return {token: index for index, token in enumerate(values)}


def event(position, pitch, duration=24, voice=0):
    return [f"<POS_{position}>", "<ROLE_PIANO>", f"<PITCH_{pitch}>",
            f"<DUR_{duration}>", "<VELOCITY_BIN_9>", f"<VOICE_ID_{voice}>"]


def test_hard_constraints_require_each_attack_pitch_and_duration_before_eos():
    grammar = PianoTokenGrammar(vocabulary(), source_score(), melody_mode="hard")
    state = grammar.validate(["<BAR_1>", *event(0, 84), *event(24, 62), "<EOS>"])
    assert state.required_events == 2
    assert state.forced_steps == 0
    with pytest.raises(GenerationError, match="Invalid token"):
        grammar.validate(["<BAR_1>", *event(0, 60, duration=12), *event(24, 74), "<EOS>"])
    with pytest.raises(GenerationError, match="Invalid token"):
        grammar.validate(["<BAR_1>", *event(0, 72), "<EOS>"])


def test_hard_constraints_handle_polyphonic_and_repeated_source_onsets():
    solo = [NoteEvent(72, 0, 1, "solo", voice="1"),
            NoteEvent(76, 0, 2, "solo", voice="2"),
            NoteEvent(72, 1, 1, "solo", voice="1")]
    grammar = PianoTokenGrammar(vocabulary(), source_score(solo), melody_mode="hard")
    state = grammar.validate(["<BAR_1>", *event(0, 72), *event(0, 64, duration=48, voice=1),
                              *event(24, 84), "<EOS>"])
    assert state.required_events == 3
    out_of_order = grammar.validate(["<BAR_1>", *event(0, 76, duration=48),
                                     *event(0, 72), *event(24, 84), "<EOS>"])
    assert out_of_order.required_events == 3
    with pytest.raises(GenerationError):
        grammar.validate(["<BAR_1>", *event(0, 72), *event(24, 72), "<EOS>"])


def test_octave_equivalent_sustained_bass_does_not_steal_melody_duration():
    grammar = PianoTokenGrammar(vocabulary(), source_score(), melody_mode="hard")
    state = grammar.validate(["<BAR_1>", *event(0, 60, duration=96),
                              *event(0, 72), *event(24, 74), "<EOS>"])
    assert state.event_count == 3
    assert state.required_events == 2


def test_none_and_soft_do_not_require_melody():
    tokens = ["<BAR_1>", *event(48, 55), "<EOS>"]
    for mode in ("none", "soft"):
        assert PianoTokenGrammar(vocabulary(), source_score(), melody_mode=mode).validate(tokens).completed
    soft = PianoTokenGrammar(vocabulary(), source_score(), melody_mode="soft")
    state = GrammarState()
    for token in ("<BAR_1>", "<POS_0>", "<ROLE_PIANO>"):
        soft.advance(state, soft.tokens[token])
    assert soft.tokens["<PITCH_55>"] in soft.allowed_ids(state)
    assert soft.soft_bias(state)[soft.tokens["<PITCH_72>"]] > 0
    assert soft.tokens["<PITCH_55>"] not in soft.soft_bias(state)


@pytest.mark.parametrize("tokens", [
    [*event(0, 72), "<EOS>"],
    ["<BAR_1>", "<POS_0>", "<EOS>"],
    ["<BAR_1>", *event(24, 72), *event(0, 74), "<EOS>"],
    ["<BAR_1>", *event(96, 72), "<EOS>"],
    ["<BAR_1>", *event(0, 72)],
    ["<BAR_1>", *event(0, 72), "<BAR_2>", "<EOS>"],
])
def test_malformed_or_out_of_form_streams_fail_explicitly(tokens):
    with pytest.raises(GenerationError):
        PianoTokenGrammar(vocabulary(), source_score(), melody_mode="none").validate(tokens)


def test_unknown_duration_and_unrepresentable_grace_requirements_fail():
    tokens = vocabulary()
    tokens.pop("<DUR_24>")
    with pytest.raises(GenerationError, match="vocabulary cannot express"):
        PianoTokenGrammar(tokens, source_score())
    grace = replace(source_score().solo[0], duration=0, is_grace=True)
    with pytest.raises(GenerationError, match="grace"):
        PianoTokenGrammar(vocabulary(), source_score([grace]))


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -.1, True])
def test_soft_bias_strength_must_be_finite(strength):
    with pytest.raises(ValueError, match="finite"):
        PianoTokenGrammar(vocabulary(), source_score(), soft_strength=strength)
