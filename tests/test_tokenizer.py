import json

import pytest

from src.representation.detokenizer import detokenize_score, detokenize_tokens
from src.representation.events import (
    DynamicEvent, MeasureInfo, NoteEvent, PartInfo, ScoreData, SignatureEvent,
    SourceScore, TempoEvent, TupletInfo, note_sort_key,
)
from src.representation.tokenizer import tokenize_score, tokenize_source


def score_fixture():
    notes = [
        NoteEvent(72, 0, 1 / 3, "solo", beat=4, role="SOLO", velocity=91,
                  pitch_name="C5", tuplets=(TupletInfo(3, 2),),
                  articulations=("staccato",), slur_ids=("slur1",)),
        NoteEvent(76, 1 / 3, 2 / 3, "solo", beat=4 + 1 / 3, role="SOLO"),
        NoteEvent(48, 0, 6, "piano", role="ACC", voice="bass", velocity=64),
        NoteEvent(60, 0, 1, "piano", role="ACC", voice="inner", velocity=65),
        NoteEvent(72, 1, .5, "solo", measure=2, role="SOLO", beat=1),
        NoteEvent(69, 4, 1.5, "solo", measure=3, role="SOLO", beat=1),
    ]
    return ScoreData(tuple(sorted(notes, key=note_sort_key)),
                     (PartInfo("solo", "Violin", "Violin", 40), PartInfo("piano", "Piano", "Piano", 0)),
                     (MeasureInfo(1, 0, 1, "4/4", "0", 3),
                      MeasureInfo(2, 1, 3, "3/4", "1"),
                      MeasureInfo(3, 4, 2.5, "5/8", "2")),
                     (SignatureEvent(0, "4/4"), SignatureEvent(1, "3/4"), SignatureEvent(4, "5/8")),
                     (SignatureEvent(0, "D minor"),), (TempoEvent(0, 88),),
                     (DynamicEvent(0, "p", "piano"),), title="Original test miniature", composer="Test")


def test_archival_roundtrip_preserves_polyphony_pickups_meters_and_long_ties():
    score = score_fixture()
    bundle = json.loads(json.dumps(tokenize_score(score)))
    assert "<DUR_144>" in bundle["tokens"]
    assert "<DUR_8>" in bundle["tokens"]
    assert detokenize_score(bundle) == score


def test_separate_role_streams_are_reconstructable():
    score = score_fixture()
    source = SourceScore(score, tuple(n for n in score.notes if n.part_id == "solo"),
                         tuple(n for n in score.notes if n.part_id == "piano"), ("solo",), ("piano",))
    streams = tokenize_source(source)
    assert all(note.role == "SOLO" for note in detokenize_score(streams["source_solo"]).notes)
    assert all(note.role == "ACC" for note in detokenize_score(streams["source_acc"]).notes)
    assert sorted(detokenize_score(streams["source_solo"]).notes + detokenize_score(streams["source_acc"]).notes,
                  key=note_sort_key) == list(score.notes)


def test_unrepresentable_tuplet_rejected_instead_of_rounded():
    score = score_fixture().with_notes([NoteEvent(60, 0, .2, "piano", tuplets=(TupletInfo(5, 4),))])
    with pytest.raises(ValueError, match="not representable"):
        tokenize_score(score)
    assert detokenize_score(tokenize_score(score, subdivisions=120)) == score


def test_grace_notes_require_archival_metadata():
    score = score_fixture().with_notes([NoteEvent(72, 0, 0, "solo", is_grace=True)])
    bundle = tokenize_score(score)
    assert detokenize_score(bundle) == score
    with pytest.raises(ValueError, match="grace"):
        detokenize_tokens(bundle["tokens"])


@pytest.mark.parametrize("tokens", [
    ["<BAR_1>"],
    ["<POS_0>", "<ROLE_PIANO>", "<PITCH_60>", "<DUR_24>", "<VELOCITY_BIN_9>", "<VOICE_ID_0>"],
    ["<BAR_1>", "<POS_0>", "<ROLE_PIANO>", "<PITCH_128>", "<DUR_24>", "<VELOCITY_BIN_9>", "<VOICE_ID_0>"],
    ["<BAR_1>", "<POS_0>", "<ROLE_PIANO>", "<PITCH_60>", "<DUR_24>", "<VELOCITY_BIN_16>", "<VOICE_ID_0>"],
    ["<BAR_1>", "<POS_96>", "<ROLE_PIANO>", "<PITCH_60>", "<DUR_24>", "<VELOCITY_BIN_9>", "<VOICE_ID_0>"],
    ["<BAR_1>", "<POS_0>", "<PITCH_60>", "<ROLE_PIANO>", "<DUR_24>", "<VELOCITY_BIN_9>", "<VOICE_ID_0>"],
    ["<BAR_1>", "<POS_0>", "<ROLE_PIANO>", "<PITCH_60>", "<DUR_-1>", "<VELOCITY_BIN_9>", "<VOICE_ID_0>"],
    ["<BAR_1>", "<POS_0>", "<ROLE_PIANO>", "<PITCH_60>", "<DUR_24>", "<VELOCITY_BIN_9>"],
])
def test_malformed_streams_fail_explicitly(tokens):
    with pytest.raises(ValueError):
        detokenize_tokens(tokens)


def test_decreasing_positions_and_missing_voices_fail():
    bundle = tokenize_score(score_fixture())
    tokens = ["<BAR_1>", "<POS_12>", "<ROLE_PIANO>", "<PITCH_60>", "<DUR_12>", "<VELOCITY_BIN_9>", "<VOICE_ID_0>",
              "<POS_0>", "<ROLE_PIANO>", "<PITCH_64>", "<DUR_12>", "<VELOCITY_BIN_9>", "<VOICE_ID_0>"]
    with pytest.raises(ValueError, match="nondecreasing"):
        detokenize_tokens(tokens)
    bundle["voice_table"] = []
    with pytest.raises(ValueError, match="VOICE_ID"):
        detokenize_score(bundle)


def test_empty_score_metadata_survives_roundtrip():
    score = score_fixture().with_notes([])
    assert detokenize_score(tokenize_score(score)) == score


def test_note_measure_mismatch_is_rejected():
    score = score_fixture().with_notes([NoteEvent(60, 1, 1, "piano", measure=1)])
    with pytest.raises(ValueError, match="outside"):
        tokenize_score(score)


def test_bare_tokens_use_notated_compound_meter_beats():
    score = ScoreData((NoteEvent(60, .5, .5, "piano"),
                       NoteEvent(62, 1.5, .5, "piano")),
                      (PartInfo("piano", "Piano", "Piano", 0),),
                      (MeasureInfo(1, 0, 3, "6/8"),))
    notes = detokenize_tokens(tokenize_score(score)["tokens"], measures=score.measures)
    assert notes[0].beat == pytest.approx(4 / 3)
    assert notes[1].beat == 2


def test_token_roundtrip_accepts_additive_meter():
    score = ScoreData((NoteEvent(60, 1.5, .5, "piano", beat=2),),
                      (PartInfo("piano", "Piano", "Piano", 0),),
                      (MeasureInfo(1, 0, 2.5, "3/8+2/8"),))
    assert detokenize_score(tokenize_score(score)) == score
    notes = detokenize_tokens(tokenize_score(score)["tokens"], measures=score.measures)
    assert notes[0].beat == 2
