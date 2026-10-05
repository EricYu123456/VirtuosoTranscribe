"""Consistent notated beat coordinates, including compound meter and pickups."""

from music21 import meter
import math

from src.representation.events import MeasureInfo


def beat_position(measure: MeasureInfo, onset: float) -> float:
    signature = meter.TimeSignature(measure.time_signature)
    local = onset - measure.onset + measure.padding_left
    nominal = float(signature.barDuration.quarterLength)
    # Imported overfull measures retain their true timeline. Extend the notated
    # beat coordinate instead of letting music21 raise at the next bar boundary.
    cycles = math.floor(local / nominal) if local >= nominal else 0
    return cycles * signature.beatCount + float(signature.getBeatProportion(local - cycles * nominal))
