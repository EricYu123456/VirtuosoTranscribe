import pytest

from src.baselines.rule_based import ArrangementControls, arrange_rule_based
from src.data.augment import transpose_pair
from src.data.mine_pairs import metadata_license_eligible, mine_pairs, normalize_title
from src.data.normalize_score import normalize_source
from src.data.parse_musicxml import parse_score
from src.representation.events import MeasureInfo, NoteEvent, PartInfo, ScoreData
from src.representation.timing import beat_position


def test_figuration_rewrites_inner_chord_without_changing_melody_or_bass():
    score = ScoreData((NoteEvent(72, 0, 4, "solo"), NoteEvent(48, 0, 4, "acc"),
                       NoteEvent(52, 0, 4, "acc"), NoteEvent(55, 0, 4, "acc")),
                      (PartInfo("solo", "Violin", "Violin", 40), PartInfo("acc", "Piano", "Piano", 0)),
                      (MeasureInfo(1, 0, 4),))
    source = normalize_source(score)
    plain = arrange_rule_based(source, ArrangementControls(inner_voice_density=1, texture_density=1))
    figured = arrange_rule_based(source, ArrangementControls(figuration_density=1, inner_voice_density=1, texture_density=1))
    assert any(n.onset == 0 and n.duration == 4 and n.pitch == 48 for n in figured.notes)
    assert any(n.onset == 0 and n.duration == 4 and n.pitch == 72 for n in figured.notes)
    assert len(figured.notes) > len(plain.notes)
    assert any(n.onset == .5 and n.pitch % 12 in {4, 7} for n in figured.notes)


def test_pair_transposition_preserves_timing_and_rejects_invalid_range():
    source = parse_score("examples/source.musicxml")
    target = parse_score("examples/target.musicxml")
    a, b = transpose_pair(source, target, 5)
    assert [n.pitch for n in a.notes] == [n.pitch + 5 for n in source.notes]
    assert [n.onset for n in b.notes] == [n.onset for n in target.notes]
    with pytest.raises(ValueError, match="range"):
        transpose_pair(source, target, 100)


def test_mining_never_approves_candidates_or_uses_conflicting_licenses():
    common = {"song_name": "Study (piano arrangement)", "composer_name": "Original Composer",
              "song_length.beats": "16", "license": "cc0", "license_conflict": "False",
              "has_paywall": "False", "is_official": "False", "subset:all_valid": "True"}
    source = dict(common, tracks="40-0", mxl="source.mxl", metadata="source.json")
    target = dict(common, tracks="0-0", mxl="target.mxl", metadata="target.json")
    candidates = mine_pairs([source, target])
    assert len(candidates) == 1
    assert candidates[0].review_status == "NEEDS_REVIEW"
    assert candidates[0].confidence == 1
    assert mine_pairs([source, dict(target, license_conflict="True")]) == []
    assert not metadata_license_eligible(dict(target, license="allrightsreserved"))
    assert metadata_license_eligible(dict(target, license="cc-zero"))
    assert not metadata_license_eligible(dict(target, license_conflict="unknown"))
    assert not metadata_license_eligible(dict(target, has_paywall="unknown"))
    assert normalize_title("Ständchen (Piano arrangement)") == "standchen"


def test_overfull_import_retains_timing_with_explicit_warning(tmp_path):
    from music21 import meter, note, stream
    score = stream.Score()
    part = stream.Part()
    measure = stream.Measure(number=1)
    measure.insert(0, meter.TimeSignature("4/4"))
    measure.insert(0, note.Note("C4", quarterLength=4))
    measure.insert(4, note.Note("G4", quarterLength=.5))
    part.append(measure)
    score.insert(0, part)
    path = tmp_path / "overfull.musicxml"
    score.write("musicxml", fp=path)
    parsed = parse_score(path)
    assert parsed.measures[0].duration == 4.5
    assert parsed.notes[1].onset == 4
    assert parsed.notes[1].beat == 5
    assert any("Overfull measure" in warning for warning in parsed.warnings)
    assert beat_position(MeasureInfo(1, 0, 3.5, "6/8"), 3) == 3


def test_lead_sheet_chord_symbol_is_not_a_zero_duration_sounding_chord(tmp_path):
    from music21 import harmony, meter, note, stream
    score = stream.Score()
    part = stream.Part()
    measure = stream.Measure(number=1)
    measure.insert(0, meter.TimeSignature("4/4"))
    measure.insert(0, harmony.ChordSymbol("C"))
    measure.insert(0, note.Note("G4", quarterLength=4))
    part.append(measure)
    score.insert(0, part)
    path = tmp_path / "lead_sheet.musicxml"
    score.write("musicxml", fp=path)
    parsed = parse_score(path)
    assert [(n.pitch, n.duration) for n in parsed.notes] == [(67, 4)]
    assert any("Chord symbols" in warning for warning in parsed.warnings)
