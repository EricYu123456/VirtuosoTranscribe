"""Conservative concert-pitch transposition that retains pair/work split identity."""

from dataclasses import replace

from src.representation.events import ScoreData, SignatureEvent


def transpose_score(score: ScoreData, semitones: int, *, piano_range: bool = False) -> ScoreData:
    if isinstance(semitones, bool) or not isinstance(semitones, int):
        raise ValueError("Transposition must be an integer number of semitones")
    lower, upper = (21, 108) if piano_range else (0, 127)
    if any(not lower <= note.pitch + semitones <= upper for note in score.notes):
        raise ValueError("Transposition moves notes outside the permitted instrument range")
    signatures = []
    from music21 import key
    for event in score.key_signatures:
        sharps, _, mode = event.value.partition(":")
        transposed = key.KeySignature(int(sharps)).asKey(mode or "major").transpose(semitones)
        signatures.append(SignatureEvent(event.onset, f"{transposed.sharps}:{mode}"))
    return score.with_notes([replace(note, pitch=note.pitch + semitones, pitch_name=None)
                             for note in score.notes], key_signatures=tuple(signatures))


def transpose_pair(source: ScoreData, target: ScoreData, semitones: int) -> tuple[ScoreData, ScoreData]:
    return transpose_score(source, semitones), transpose_score(target, semitones, piano_range=True)
