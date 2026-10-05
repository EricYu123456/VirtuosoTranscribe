from dataclasses import replace

import pytest

from src.baselines._piano import can_add, overlaps
from src.baselines.literal_reduction import reduce_literal
from src.baselines.rule_based import ArrangementControls, arrange_rule_based, load_preset
from src.inference.generate import generate_candidates
from src.representation.events import DynamicEvent, MeasureInfo, NoteEvent, PartInfo, ScoreData, SourceScore


def make_source(solo=None, accompaniment=None):
    solo = tuple(solo if solo is not None else [NoteEvent(72, i, 1, "solo", role="SOLO") for i in range(4)])
    accompaniment = tuple(accompaniment if accompaniment is not None else [
        NoteEvent(48, i, 1, "piano", role="ACC") for i in range(4)])
    score = ScoreData(solo + accompaniment, (PartInfo("solo", "Violin", "Violin"),
                     PartInfo("piano", "Piano", "Piano")),
                     (MeasureInfo(1, 0, 4),),
                     dynamics=(DynamicEvent(0, "p", "piano"),), title="Test")
    return SourceScore(score, solo, accompaniment, ("solo",), ("piano",))


def test_literal_preserves_solo_and_bass_and_piano_parts():
    source = make_source()
    result = reduce_literal(source)
    melody = [n for n in result.notes if n.role == "SOLO"]
    assert [(n.pitch, n.onset, n.duration) for n in melody] == [(n.pitch, n.onset, n.duration) for n in source.solo]
    assert [(n.pitch, n.onset) for n in result.notes if n.role == "ACC"] == [(48, i) for i in range(4)]
    assert [p.id for p in result.parts] == ["piano_rh", "piano_lh"]
    assert all(p.instrument == "Piano" for p in result.parts)
    assert result.dynamics[0].part_id == "piano_lh"


def test_literal_handles_sustained_unison_collisions():
    source = make_source(solo=[NoteEvent(72, 1, 1, "solo")],
                         accompaniment=[NoteEvent(72, 0, 4, "piano"), NoteEvent(48, 0, 4, "piano")])
    result = reduce_literal(source)
    for index, note in enumerate(result.notes):
        for other in result.notes[index + 1:]:
            assert note.pitch != other.pitch or not overlaps(note, other)
    assert all(21 <= n.pitch <= 108 for n in result.notes)
    assert any(n.pitch % 12 == 0 and n.onset == 0 and n.duration == 4 for n in result.notes)


@pytest.mark.parametrize("reduce", [reduce_literal, arrange_rule_based])
def test_baselines_merge_identical_unison_events(reduce):
    solo = NoteEvent(72, 0, 4, "solo")
    accompaniment = NoteEvent(48, 0, 4, "piano")
    result = reduce(make_source(solo=[solo, replace(solo, voice="2")],
                                accompaniment=[accompaniment, replace(accompaniment, voice="2")]))
    events = [(n.pitch, n.onset, n.duration) for n in result.notes]
    assert len(events) == len(set(events)) == 2
    assert any("unison" in message for message in result.warnings)


def test_octave_doubling_control_is_effective_and_preserves_rhythm():
    source = make_source()
    plain = arrange_rule_based(source, ArrangementControls(octave_doubling=0))
    doubled = arrange_rule_based(source, ArrangementControls(octave_doubling=1))
    assert sum(n.role == "MELODY_DOUBLE" for n in plain.notes) == 0
    assert sum(n.role == "MELODY_DOUBLE" for n in doubled.notes) == len(source.solo)
    for original in source.solo:
        assert any(n.role == "SOLO" and n.pitch % 12 == original.pitch % 12
                   and n.onset == original.onset and n.duration == original.duration for n in doubled.notes)


def test_rule_based_caps_sustained_hand_span_and_density():
    accompaniment = [NoteEvent(p, i / 4, 4 - i / 4, "piano")
                     for i, p in enumerate([21, 32, 43, 54, 65, 76, 87, 98, 108])]
    source = make_source(solo=[NoteEvent(84, 0, 4, "solo")], accompaniment=accompaniment)
    result = arrange_rule_based(source, ArrangementControls(texture_density=.35, inner_voice_density=1))
    for time in sorted({n.onset for n in result.notes}):
        active = [n for n in result.notes if n.onset <= time < n.end]
        assert len(active) <= 6
        for hand in ("piano_rh", "piano_lh"):
            pitches = [n.pitch for n in active if n.part_id == hand]
            assert len(pitches) <= 4
            if pitches:
                assert max(pitches) - min(pitches) <= 12
    assert any(n.role == "ACC" and n.pitch % 12 == 21 % 12 and n.onset == 0 for n in result.notes)
    assert all(21 <= n.pitch <= 108 for n in result.notes)


def test_rule_based_protects_upward_bass_motion_under_held_predecessor():
    source = make_source(solo=[NoteEvent(84, 0, 4, "solo")], accompaniment=[
        NoteEvent(48, 0, 4, "piano"), NoteEvent(50, 1, 2, "piano"),
        NoteEvent(52, 2, 1, "piano")])
    result = arrange_rule_based(source, ArrangementControls(inner_voice_density=0))
    for original in source.accompaniment:
        assert any(n.role == "ACC" and n.pitch % 12 == original.pitch % 12
                   and n.onset == original.onset and n.duration == original.duration for n in result.notes)


def test_interval_allocator_does_not_combine_disjoint_held_notes():
    allocated = [NoteEvent(60, 0, 1, "piano_rh", role="ACC"),
                 NoteEvent(84, 1, 1, "piano_rh", role="ACC")]
    candidate = NoteEvent(72, 0, 2, "piano_rh", role="ACC")
    assert can_add(candidate, allocated, max_span=12, max_per_hand=2)
    assert not can_add(replace(candidate, pitch=73), allocated, max_span=12)


@pytest.mark.parametrize("value", [-.01, 1.01, float("nan"), float("inf"), True])
def test_controls_reject_out_of_range_values(value):
    with pytest.raises(ValueError, match="finite number"):
        ArrangementControls(octave_doubling=value)


@pytest.mark.parametrize("name", ["harmonic_freedom", "rhythmic_freedom", "countermelody", "ornamentation"])
def test_unsupported_controls_are_explicit(name):
    with pytest.raises(ValueError, match="V1"):
        ArrangementControls(**{name: .1})
    with pytest.raises(ValueError, match="melody_preservation"):
        ArrangementControls(melody_preservation=.5)


def test_presets_match_checked_in_configs():
    from pathlib import Path
    for name in ("faithful", "romantic", "virtuosic", "concert_transcription"):
        assert load_preset(name) == load_preset(Path("configs/presets") / f"{name}.yaml")


def test_seeded_candidates_repeat_and_can_be_distinct():
    source = make_source(solo=[NoteEvent(72, i, 1, "solo") for i in range(8)],
                         accompaniment=[NoteEvent(48, i, 1, "piano") for i in range(8)])
    controls = ArrangementControls(octave_doubling=.5)
    first = generate_candidates(source, controls, num_candidates=3, seed=5)
    second = generate_candidates(source, controls, num_candidates=3, seed=5)
    assert first == second
    assert len({tuple((n.pitch, n.onset, n.duration, n.role) for n in c.notes) for c in first}) == 3


def test_candidate_fingerprints_ignore_technique_labels():
    from src.inference.generate import _fingerprint
    score = arrange_rule_based(make_source(), ArrangementControls(octave_doubling=1))
    relabeled = score.with_notes([replace(n, role="ACC") if n.role == "MELODY_DOUBLE" else n
                                 for n in score.notes])
    assert _fingerprint(score) == _fingerprint(relabeled)


def test_empty_solo_and_invalid_candidate_counts_fail():
    source = make_source(solo=[])
    with pytest.raises(ValueError, match="solo"):
        reduce_literal(source)
    with pytest.raises(ValueError, match="solo"):
        arrange_rule_based(source)
    with pytest.raises(ValueError, match="positive integer"):
        generate_candidates(make_source(), num_candidates=0)


def test_reranking_prefers_preserved_melody():
    from src.inference.rerank import rank_candidates
    source = make_source()
    preserved = arrange_rule_based(source)
    missing = preserved.with_notes([n for n in preserved.notes if n.role != "SOLO"])
    ranked = rank_candidates(source, [missing, preserved])
    assert ranked[0]["candidate_index"] == 1
    assert ranked[0]["metrics"]["melody_coverage_octave_tolerant"] == 1
    assert ranked[0]["score"] > ranked[1]["score"]


def test_generation_cli_exports_grand_staff_midi_and_ranked_analysis(tmp_path):
    import json
    import xml.etree.ElementTree as ET
    from src.inference.export import export_score
    from src.inference.generate import main
    source = make_source()
    input_path = tmp_path / "source.musicxml"
    export_score(source.score, input_path)
    output_dir = tmp_path / "generated"
    assert main(["--input", str(input_path), "--preset", "romantic",
                 "--num_candidates", "2", "--output_dir", str(output_dir),
                 "--solo-part", "Violin", "--accompaniment-part", "Piano"]) == 0
    ranking = json.loads((output_dir / "ranking.json").read_text(encoding="utf-8"))
    assert ranking["neural_model"] is False
    assert ranking["generator"] == "rule_based"
    assert len(ranking["ranking"]) == 2
    for stem in ("candidate_01", "candidate_02", "best"):
        assert (output_dir / f"{stem}.mid").read_bytes().startswith(b"MThd")
        musicxml = ET.parse(output_dir / f"{stem}.musicxml")
        assert len(musicxml.findall("./part-list/score-part")) == 1
        assert musicxml.findtext("./part/measure/attributes/staves") == "2"
    stats = json.loads((output_dir / "analysis_01.json").read_text(encoding="utf-8"))
    assert stats["melody_coverage_octave_tolerant"] == 1
    assert "<svg" in (output_dir / "alignment.html").read_text(encoding="utf-8")
    assert json.loads((output_dir / "alignment.json").read_text(encoding="utf-8"))["quality"] > 0
    assert (output_dir / "source.json").is_file()


def test_generation_cli_errors_are_clean(tmp_path, capsys):
    from src.inference.export import export_score
    from src.inference.generate import main
    input_path = tmp_path / "source.musicxml"
    export_score(make_source().score, input_path)
    with pytest.raises(SystemExit) as error:
        main(["--input", str(input_path), "--output_dir", str(tmp_path / "output"),
              "--num_candidates", "0"])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "positive integer" in stderr
    assert "Traceback" not in stderr

