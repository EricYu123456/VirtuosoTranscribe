from dataclasses import replace
from pathlib import Path
import wave

import numpy as np
import pytest

from src.inference.audio import TempoMap, render_audio
from src.representation.events import NoteEvent, PartInfo, ScoreData, TempoEvent


def simple_score():
    return ScoreData(notes=(NoteEvent(60, 0, 4, "piano", velocity=80),),
                     parts=(PartInfo("piano", "Piano"),), measures=(),
                     tempos=(TempoEvent(0, 60), TempoEvent(2, 120)))


def read_wav(path):
    with wave.open(str(path), "rb") as source:
        return source.getframerate(), np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")


def test_audio_integrates_tempo_changes_inside_a_sustained_note(tmp_path):
    score = simple_score()
    tempo = TempoMap(score)
    assert tempo.seconds_at(1) == 1
    assert tempo.seconds_at(2) == 2
    assert tempo.seconds_at(4) == 3
    metadata = render_audio(score, tmp_path / "preview.wav", sample_rate=8000, release=.25)
    rate, samples = read_wav(tmp_path / "preview.wav")
    assert rate == 8000
    assert len(samples) == 26000
    assert metadata["score_duration_seconds"] == 3
    assert np.max(np.abs(samples)) > 0
    assert abs(int(samples[-1])) <= 1


def test_polyphonic_audio_does_not_overflow_and_preserves_relative_velocity(tmp_path):
    score = replace(simple_score(), notes=tuple(NoteEvent(48 + i, 0, 4, "piano", velocity=127) for i in range(40)))
    render_audio(score, tmp_path / "dense.wav", sample_rate=8000)
    _, samples = read_wav(tmp_path / "dense.wav")
    assert np.max(np.abs(samples.astype(np.int32))) <= 31130
    soft = replace(simple_score(), notes=(NoteEvent(60, 0, 4, "piano", velocity=25),))
    render_audio(soft, tmp_path / "soft.wav", sample_rate=8000)
    render_audio(simple_score(), tmp_path / "loud.wav", sample_rate=8000)
    _, quiet = read_wav(tmp_path / "soft.wav")
    _, loud = read_wav(tmp_path / "loud.wav")
    assert np.std(quiet.astype(float)) < np.std(loud.astype(float)) / 2


@pytest.mark.parametrize("kwargs", [{"sample_rate": 100}, {"sample_rate": True}, {"release": -1},
                                    {"max_duration": 2}, {"engine": "mystery"}, {"engine": "fluidsynth"}])
def test_audio_validation_never_silently_truncates(tmp_path, kwargs):
    with pytest.raises(ValueError):
        render_audio(simple_score(), tmp_path / "invalid.wav", **kwargs)
    assert not (tmp_path / "invalid.wav").exists()


def test_bad_tempo_and_destination_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="Tempo BPM"):
        TempoMap(replace(simple_score(), tempos=(TempoEvent(0, 0),)))
    with pytest.raises(ValueError, match="extension"):
        render_audio(simple_score(), Path(tmp_path) / "not.wav.exe")
