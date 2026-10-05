"""Self-contained HTML/SVG alignment piano roll with per-note hover details."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

from src.representation.events import NoteEvent, ScoreData
from .melody_alignment import AlignmentResult


def write_alignment_visualization(source_solo: Iterable[NoteEvent],
                                  target: ScoreData | Iterable[NoteEvent],
                                  alignment: AlignmentResult, path: str | Path,
                                  *, title: str = "Melody alignment") -> Path:
    source = tuple(source_solo)
    target_notes = target.notes if isinstance(target, ScoreData) else tuple(target)
    all_notes = source + target_notes
    duration = max((n.end for n in all_notes), default=1.0)
    low, high = min((n.pitch for n in all_notes), default=48), max((n.pitch for n in all_notes), default=84)
    margin, width, band_height = 64, max(980, min(10000, int(duration * 55))), max(180, (high - low + 1) * 6)
    height = band_height * 2 + 120
    xscale, yscale = (width - margin - 20) / max(duration, 1), (band_height - 30) / max(high - low + 1, 1)

    def x(onset):
        return margin + onset * xscale

    def y(pitch, lower=False):
        return 45 + (band_height + 45 if lower else 0) + (high - pitch) * yscale

    source_matched = {m.source_index for m in alignment.matches}
    target_matched = {m.target_index for m in alignment.matches}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title, quote=True)}">',
           '<style>text{font:12px sans-serif;fill:#334155}.grid{stroke:#e2e8f0;stroke-width:1}.match-link{stroke:#64748b;stroke-width:1;opacity:.32}rect.note:hover{stroke:#111827;stroke-width:2}</style>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="14" y="20">SOURCE SOLO</text>',
           f'<text x="14" y="{band_height + 65}">TARGET PIANO</text>']
    # Integer quarter-note grid (not assuming a fixed meter).
    grid_step = max(1, int(duration / 250))
    for beat in range(0, int(duration) + 1, grid_step):
        xp = x(beat)
        svg.extend([f'<line class="grid" x1="{xp:.2f}" y1="30" x2="{xp:.2f}" y2="{height - 20}"/>',
                    f'<text x="{xp:.2f}" y="{height - 5}">{beat}</text>'])
    for pitch in range(low, high + 1):
        if pitch % 12 == 0:
            for lower in (False, True):
                yp = y(pitch, lower)
                svg.extend([f'<line class="grid" x1="{margin}" y1="{yp:.2f}" x2="{width - 20}" y2="{yp:.2f}"/>',
                            f'<text x="14" y="{yp + 4:.2f}">C{pitch // 12 - 1}</text>'])
    for match in alignment.matches:
        if match.source_index >= len(source) or match.target_index >= len(target_notes):
            raise ValueError("Alignment indices do not match visualization scores")
        a, b = source[match.source_index], target_notes[match.target_index]
        svg.append(f'<line class="match-link" x1="{x(a.onset):.2f}" y1="{y(a.pitch) + yscale / 2:.2f}" x2="{x(b.onset):.2f}" y2="{y(b.pitch, True) + yscale / 2:.2f}"><title>Source {match.source_index} → target {match.target_index}; shift {match.octave_shift:+d}; onset deviation {match.onset_deviation:+.4f}; cost {match.cost:.3f}</title></line>')
    for lower, notes, matched in ((False, source, source_matched), (True, target_notes, target_matched)):
        for index, note in enumerate(notes):
            color = ("#22c55e" if lower else "#3b82f6") if index in matched else ("#cbd5e1" if lower else "#ef4444")
            label = escape(f"{'Target' if lower else 'Source'} index {index}; MIDI {note.pitch}; onset {note.onset:g}; duration {note.duration:g}; measure {note.measure}; beat {note.beat:g}; {note.part_id}/{note.voice}; {'matched' if index in matched else 'unmatched'}")
            svg.append(f'<rect class="note" x="{x(note.onset):.2f}" y="{y(note.pitch, lower):.2f}" width="{max(2, note.duration * xscale):.2f}" height="{max(3, yscale - .6):.2f}" fill="{color}" rx="1"><title>{label}</title></rect>')
    svg.append('</svg>')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>body{{font:15px system-ui,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}}h1{{font-size:22px}}.roll{{overflow:auto;border:1px solid #cbd5e1;background:white}}svg{{width:{width}px;max-width:none}}p{{max-width:90ch}}button{{padding:6px 10px}}</style>
<h1>{escape(title)}</h1><p>Quality {alignment.quality:.3f} · {len(alignment.matches)} matched · {len(alignment.unmatched_source_indices)} omitted source notes · onset tolerance {alignment.onset_tolerance:g} quarter notes. Hover over notes and alignment lines for indices and timing.</p>
<p>Blue: aligned source. Red: unmatched source. Green: aligned target. Gray: other target notes. Horizontal axis: quarter notes. Octave relocation is accepted. Simultaneous notes are matched one to one.</p>
<button onclick="document.querySelectorAll('.match-link').forEach(el=>el.style.display=el.style.display==='none'?'':'none')">Toggle alignment lines</button>
<div class="roll">{''.join(svg)}</div><p>Inspect this alignment before using it as training supervision. Same-pitch accompaniment can be mistaken for a melody; this plot does not establish artistic fidelity.</p></html>'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
