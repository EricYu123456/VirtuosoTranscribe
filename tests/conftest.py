"""Hand-verifiable notation fixture covering pickup, ties, and tuplets."""

from pathlib import Path

from music21 import (
    articulations, chord, dynamics, duration, instrument, key, meter, note,
    spanner, stream, tempo, tie,
)
import pytest


@pytest.fixture
def notation_path(tmp_path: Path) -> Path:
    score = stream.Score()
    solo, piano = stream.Part(id="cello"), stream.Part(id="piano")
    solo.partName, piano.partName = "Violoncello", "Piano"
    solo.insert(0, instrument.Violoncello())
    piano.insert(0, instrument.Piano())
    slur_members = []
    for index, length in enumerate((1, 4, 3)):
        for part in (solo, piano):
            measure = stream.Measure(number=index)
            if index == 0:
                measure.insert(0, meter.TimeSignature("4/4"))
                measure.insert(0, key.KeySignature(-1))
                measure.insert(0, tempo.MetronomeMark(number=90))
                measure.paddingLeft = 3
            if index == 2:
                measure.insert(0, meter.TimeSignature("3/4"))
                measure.insert(0, key.KeySignature(1))
            if part is solo:
                if index == 0:
                    element = note.Note("F4", quarterLength=1)
                    element.tie = tie.Tie("start")
                    element.articulations = [articulations.Tenuto()]
                    measure.insert(0, dynamics.Dynamic("p"))
                    measure.append(element)
                    slur_members.append(element)
                elif index == 1:
                    element = note.Note("F4", quarterLength=1)
                    element.tie = tie.Tie("stop")
                    measure.append(element)
                    for pitch in ("G4", "A4", "B-4"):
                        element = note.Note(pitch)
                        element.duration = duration.Duration("eighth")
                        element.duration.appendTuplet(duration.Tuplet(3, 2))
                        measure.append(element)
                    slur_members.append(element)
                    measure.append(note.Rest(quarterLength=2))
                else:
                    measure.append(note.Note("G4", quarterLength=3))
            else:
                measure.append(chord.Chord(("C3", "F3"), quarterLength=length))
            part.append(measure)
    solo.insert(0, spanner.Slur(*slur_members))
    score.insert(0, solo)
    score.insert(0, piano)
    path = tmp_path / "notation.musicxml"
    score.write("musicxml", fp=str(path))
    return path
