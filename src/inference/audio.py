"""Portable listening previews and an explicit optional FluidSynth renderer.

The built-in additive renderer is a synthetic piano-like timbre, not a sampled
piano or expressive performance. Quarter-note timing follows all tempo changes.
"""

from __future__ import annotations

from bisect import bisect_right
import math
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np

from src.representation.events import ScoreData


class TempoMap:
    """Piecewise integration of quarter-note beats into elapsed seconds."""

    def __init__(self, score: ScoreData, default_bpm: float = 120.0):
        if not math.isfinite(default_bpm) or default_bpm <= 0:
            raise ValueError("Default tempo must be positive and finite")
        events = {0.0: float(default_bpm)}
        for event in score.tempos:
            if not math.isfinite(event.onset) or event.onset < 0:
                raise ValueError("Tempo onset must be non-negative and finite")
            if not math.isfinite(event.bpm) or event.bpm <= 0:
                raise ValueError("Tempo BPM must be positive and finite")
            events[float(event.onset)] = float(event.bpm)
        self.beats = sorted(events)
        self.bpms = [events[b] for b in self.beats]
        self.seconds = [0.0]
        for i in range(1, len(self.beats)):
            self.seconds.append(self.seconds[-1] +
                                (self.beats[i] - self.beats[i - 1]) * 60 / self.bpms[i - 1])

    def seconds_at(self, beat: float) -> float:
        if not math.isfinite(beat) or beat < 0:
            raise ValueError("Beat must be non-negative and finite")
        index = bisect_right(self.beats, beat) - 1
        return self.seconds[index] + (beat - self.beats[index]) * 60 / self.bpms[index]


def render_audio(score: ScoreData, path: str | Path, *, sample_rate: int = 22050,
                 max_duration: float = 600.0, release: float = 0.25,
                 engine: str = "synthetic", soundfont: str | Path | None = None,
                 fluidsynth: str = "fluidsynth") -> dict:
    """Write PCM WAV, returning measured metadata; never silently truncate a score."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
        raise ValueError("sample_rate must be an integer in 8000..96000")
    if not math.isfinite(max_duration) or max_duration <= 0:
        raise ValueError("max_duration must be positive and finite")
    if not math.isfinite(release) or not 0 <= release <= 5:
        raise ValueError("release must be finite and in 0..5 seconds")
    tempo = TempoMap(score)
    duration = tempo.seconds_at(score.duration)
    audible_end = max([duration] + [tempo.seconds_at(n.onset) + .06 for n in score.notes if n.is_grace])
    if audible_end + release > max_duration:
        raise ValueError(f"Audio duration {audible_end + release:.2f}s exceeds limit {max_duration:g}s")
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise ValueError("Audio output must have a .wav extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    if engine == "fluidsynth":
        if soundfont is None or not Path(soundfont).is_file():
            raise ValueError("FluidSynth requires an existing user-configured SoundFont")
        from src.inference.export import export_score
        with tempfile.TemporaryDirectory(prefix="virtuoso_audio_") as directory:
            midi = export_score(score, Path(directory) / "performance.mid")
            try:
                subprocess.run([fluidsynth, "-ni", "-F", str(path.resolve()), "-r", str(sample_rate),
                                str(Path(soundfont).resolve()), str(midi)],
                               check=True, capture_output=True, text=True,
                               timeout=min(600, max(30, duration * 2)))
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError(f"FluidSynth rendering failed: {exc}") from exc
        if not path.is_file():
            raise ValueError("FluidSynth did not produce its requested WAV file")
        with wave.open(str(path), "rb") as stream:
            actual_duration = stream.getnframes() / stream.getframerate()
            if actual_duration > max_duration:
                path.unlink()
                raise ValueError("FluidSynth output exceeds the audio duration limit")
            return {"path": str(path.resolve()), "engine": engine,
                    "sample_rate": stream.getframerate(), "channels": stream.getnchannels(),
                    "duration_seconds": actual_duration, "score_duration_seconds": duration,
                    "quality": "SoundFont synthesis; no expressive timing or pedal prediction"}
    if engine != "synthetic":
        raise ValueError("Audio engine must be synthetic or fluidsynth")
    count = max(1, math.ceil((audible_end + release) * sample_rate))
    samples = np.zeros(count, dtype=np.float32)
    for note in score.notes:
        start = tempo.seconds_at(note.onset)
        stop = tempo.seconds_at(note.end)
        held = max(0.06 if note.is_grace else 1 / sample_rate, stop - start)
        index = round(start * sample_rate)
        size = min(count - index, math.ceil((held + release) * sample_rate))
        if size <= 0:
            continue
        t = np.arange(size, dtype=np.float32) / sample_rate
        frequency = 440 * 2 ** ((note.pitch - 69) / 12)
        signal = np.zeros(size, dtype=np.float32)
        for harmonic in range(1, 9):
            if frequency * harmonic >= sample_rate / 2:
                break
            signal += (np.sin(2 * np.pi * frequency * harmonic * t) / harmonic ** 1.65)
        attack = np.minimum(t / 0.006, 1)
        # A struck string decays during its notated hold. Release starts at the
        # integrated note-off, even when a tempo change occurs inside the note.
        envelope = attack * np.exp(-t * (1.3 + frequency / 2200))
        if release:
            envelope *= np.where(t <= held, 1, np.maximum(0, 1 - (t - held) / release) ** 2)
        else:
            envelope *= t <= held
        samples[index:index + size] += signal * envelope * (note.velocity / 127) ** 1.6 * 0.18
    peak = float(np.max(np.abs(samples)))
    if peak > 0.95:
        samples *= 0.95 / peak
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return {"path": str(path.resolve()), "engine": "synthetic", "sample_rate": sample_rate,
            "channels": 1, "duration_seconds": count / sample_rate,
            "score_duration_seconds": duration, "release_seconds": release,
            "quality": "Synthetic additive piano-like listening preview; not a sampled piano performance"}
