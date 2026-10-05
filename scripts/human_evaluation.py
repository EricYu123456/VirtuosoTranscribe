"""Prepare blind listening questionnaires and aggregate only supplied listener ratings."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
from html import escape
import json
from pathlib import Path
import random
import shutil
import statistics
import uuid

from src.data.build_manifest import load_manifest
from src.data.parse_musicxml import parse_score
from src.inference.audio import render_audio
from src.inference.export import export_score
from src.representation.events import ScoreData
from scripts.run_experiments import SUMMARY_METRICS


QUESTIONS = {
    "melody_preservation": "How well is the original melody preserved?",
    "piano_naturalness": "How natural is the piano writing?",
    "coherence": "How musically coherent is the arrangement?",
    "expressiveness": "How expressive is the transcription?",
    "playability": "How playable does it appear?",
    "intentional_arrangement": "How much does it sound like an intentional piano arrangement?",
    "overall_preference": "Overall preference",
}
RATING_COLUMNS = ["listener_id", "candidate_id", *QUESTIONS, "comment"]


def _roll(score: ScoreData, path: Path, title: str) -> None:
    low = min((n.pitch for n in score.notes), default=48)
    high = max((n.pitch for n in score.notes), default=84)
    width, height = 1000, max(180, (high - low + 1) * 5 + 30)
    xs, ys = 940 / max(score.duration, 1), (height - 30) / max(1, high - low + 1)
    elements = []
    for pitch in range(low, high + 1):
        if pitch % 12 == 0:
            y = 10 + (high - pitch) * ys
            elements.append(f'<line x1="50" x2="990" y1="{y}" y2="{y}" stroke="#dae4ef"/><text x="4" y="{y + 5}" font-size="12">C{pitch // 12 - 1}</text>')
    for note in score.notes:
        elements.append(f'<rect x="{50 + note.onset * xs}" y="{10 + (high - note.pitch) * ys}" width="{max(2, note.duration * xs)}" height="{max(2, ys - .5)}" fill="#4477ac"><title>MIDI {note.pitch}; onset {note.onset:g}; duration {note.duration:g}</title></rect>')
    path.write_text(f'<!doctype html><html><meta charset="utf-8"><title>{escape(title)}</title><style>body{{font:14px system-ui;margin:12px}}svg{{width:100%;min-width:800px}}div{{overflow:auto}}</style><div><svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" aria-label="Piano roll">{"".join(elements)}</svg></div><p>Horizontal: quarter-note time; vertical: MIDI pitch. Hover notes for durations. Open the MusicXML score to inspect notation and inferred playability.</p></html>', encoding="utf-8")


def prepare_blind_evaluation(report_path: str | Path, output_dir: str | Path, *, seed: int = 0) -> dict:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    eligible = [row for row in report.get("rows", []) if row.get("status") == "ok"]
    if not eligible:
        raise ValueError("No successfully generated examples in the experiment report")
    root = Path(output_dir).resolve()
    participant, private = root / "participant", root / "private"
    if (private / "key.json").exists():
        raise ValueError("A blind package already exists here; choose a new output directory to preserve its rating key")
    participant.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    pairs = {p["pair_id"]: p for p in load_manifest(report["manifest"])}
    rng = random.Random(seed)
    rng.shuffle(eligible)
    work_labels = {}
    key = {}
    cards = []
    for row in eligible:
        pair_id = row["pair_id"]
        if pair_id not in pairs:
            raise ValueError("Experiment pair is absent from its manifest")
        if pair_id not in work_labels:
            work_labels[pair_id] = f"Excerpt {len(work_labels) + 1}"
            source = parse_score(pairs[pair_id]["source"]["path"])
            render_audio(source, participant / f"source_{len(work_labels)}.wav")
        source_index = list(work_labels).index(pair_id) + 1
        candidate_id = uuid.uuid4().hex[:12]
        label = f"Candidate {len(key) + 1}"
        score = parse_score(row["score"])
        anonymous = replace(score, title=label, composer="", warnings=())
        export_score(anonymous, participant / f"{candidate_id}.musicxml")
        audio = row.get("audio", {}).get("path")
        if audio and Path(audio).is_file():
            shutil.copyfile(audio, participant / f"{candidate_id}.wav")
        else:
            render_audio(anonymous, participant / f"{candidate_id}.wav")
        _roll(anonymous, participant / f"{candidate_id}.html", label)
        key[candidate_id] = {"method": row["method"], "pair_id": pair_id, "work_id": row["work_id"],
                             "label": label, "excerpt": work_labels[pair_id], "score": row["score"],
                             "metrics": {k: row.get("metrics", {}).get(k) for k in SUMMARY_METRICS}}
        prompts = "".join(f'<label>{escape(question)}<select data-question="{name}"><option value="">Unrated</option>{"".join(f"<option value={i}>{i}</option>" for i in range(1, 6))}</select></label>' for name, question in QUESTIONS.items())
        cards.append(f'<section data-candidate="{candidate_id}"><h2>{label} · {work_labels[pair_id]}</h2><div class="players"><div><p>Source</p><audio controls preload="none" src="source_{source_index}.wav"></audio></div><div><p>Piano arrangement</p><audio controls preload="none" src="{candidate_id}.wav"></audio></div></div><details><summary>Optional score inspection</summary><iframe title="{label} piano roll" src="{candidate_id}.html"></iframe><a href="{candidate_id}.musicxml" download>MusicXML score</a></details>{prompts}<label>Comment<textarea data-comment rows="2"></textarea></label></section>')
    key_payload = {"schema_version": 1, "report": str(Path(report_path).resolve()),
                   "seed": seed, "candidates": key, "questions": QUESTIONS}
    (private / "key.json").write_text(json.dumps(key_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (participant / "ratings_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RATING_COLUMNS)
        writer.writeheader()
        writer.writerows({"candidate_id": candidate_id} for candidate_id in key)
    # Participant HTML contains only anonymous IDs, references, questionnaires,
    # and synthetic audio. The method mapping stays outside the served directory.
    document = r'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind transcription listening evaluation</title><style>body{font:16px system-ui;color:#172536;background:#f1f5f9;max-width:1000px;margin:28px auto;padding:0 20px}section{padding:24px;background:white;border:1px solid #d7e1ed;border-radius:10px;margin:20px 0}label{display:flex;justify-content:space-between;align-items:center;gap:20px;margin:12px 0;font-size:14px}select,input,textarea,button{font:inherit;padding:8px;border:1px solid #b3c5d9;border-radius:6px}select{min-width:100px}textarea{width:50%}audio{width:100%}.players{display:grid;grid-template-columns:1fr 1fr;gap:20px}iframe{width:100%;height:250px;border:1px solid #d7e1ed}button{background:#2869b7;color:white;border:0;cursor:pointer}.hint{font-size:14px;color:#5d6e82}#message{color:#9a3e29}@media(max-width:600px){.players{grid-template-columns:1fr}label{display:block}select{margin-left:10px}}</style><h1>Blind piano transcription evaluation</h1><p>Listen to each source and its anonymous piano arrangement. Rate 1 (very poor) to 5 (excellent). Leave a question unrated if you cannot judge it. All audio uses the same synthetic piano-like renderer; it does not reproduce pianist expression.</p><p class="hint">Inspect the score before judging playability. This is a subjective assessment; no model labels or automatic metric scores are shown.</p><label>Listener pseudonym <input id="listener" required placeholder="e.g. listener_01"></label>''' + "".join(cards) + r'''<button id="download">Download my ratings CSV</button><p id="message" role="status"></p><script>
const columns=__COLUMNS__;
function quote(v){return '"'+String(v).replaceAll('"','""')+'"'}
document.getElementById('download').onclick=()=>{const listener=document.getElementById('listener').value.trim();if(!listener){document.getElementById('message').textContent='Enter a listener pseudonym first.';return}const rows=[columns];for(const section of document.querySelectorAll('section[data-candidate]')){const row={listener_id:listener,candidate_id:section.dataset.candidate,comment:section.querySelector('[data-comment]').value};let rated=false;for(const select of section.querySelectorAll('[data-question]')){row[select.dataset.question]=select.value;if(select.value)rated=true}if(rated)rows.push(columns.map(k=>row[k]||''))}if(rows.length===1){document.getElementById('message').textContent='Rate at least one question before exporting.';return}const blob=new Blob([rows.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n'],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='ratings.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('message').textContent='Your actual ratings were exported. Send this CSV to the study organizer.'};
</script></html>'''
    document = document.replace("__COLUMNS__", json.dumps(RATING_COLUMNS))
    (participant / "index.html").write_text(document, encoding="utf-8")
    return {"participant_page": str(participant / "index.html"), "private_key": str(private / "key.json"),
            "candidates": len(key), "ratings_collected": 0,
            "note": "Keep private/key.json private. Serve only the participant directory. No human judgments have been generated."}


def aggregate_ratings(key_path: str | Path, rating_paths: list[str | Path], output: str | Path) -> dict:
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))["candidates"]
    observed = set()
    ratings = []
    for path in rating_paths:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not set(RATING_COLUMNS[:-1]).issubset(reader.fieldnames or []):
                raise ValueError("Ratings CSV is missing listener, candidate, or question columns")
            for raw in reader:
                candidate = (raw.get("candidate_id") or "").strip()
                listener = (raw.get("listener_id") or "").strip()
                values = {}
                for name in QUESTIONS:
                    value = (raw.get(name) or "").strip()
                    if value:
                        if value not in {"1", "2", "3", "4", "5"}:
                            raise ValueError(f"Rating {name} must be an integer in 1..5 or blank")
                        values[name] = int(value)
                if not values:
                    continue  # A blank organizer template contains no judgments.
                if not listener or candidate not in key:
                    raise ValueError("Rated rows require a listener ID and a candidate from the private key")
                identity = listener, candidate
                if identity in observed:
                    raise ValueError("Duplicate listener/candidate ratings would double-count a judgment")
                observed.add(identity)
                ratings.append({"listener_id": listener, "candidate_id": candidate,
                                "method": key[candidate]["method"], "work_id": key[candidate]["work_id"],
                                "ratings": values, "comment": raw.get("comment", "")})
    groups = defaultdict(lambda: defaultdict(list))
    for row in ratings:
        for name, value in row["ratings"].items():
            groups[row["method"]][name].append(value)
    summary = {method: {name: {"count": len(values), "mean": statistics.fmean(values),
                              "median": statistics.median(values),
                              "histogram": {str(v): values.count(v) for v in range(1, 6)}}
                        for name, values in questions.items()}
               for method, questions in sorted(groups.items())}
    correlations = {}
    candidate_ratings = defaultdict(lambda: defaultdict(list))
    for row in ratings:
        for name, value in row["ratings"].items():
            candidate_ratings[row["candidate_id"]][name].append(value)

    def ranks(values):
        positions = defaultdict(list)
        for rank, (_, index) in enumerate(sorted((v, i) for i, v in enumerate(values)), 1):
            positions[values[index]].append(rank)
        return [statistics.fmean(positions[value]) for value in values]

    # Exploratory correlations use candidate-level listener means. Repeated
    # works are dependent, so these do not supply significance tests.
    for name in QUESTIONS:
        for metric in SUMMARY_METRICS:
            points = []
            for candidate, questions in candidate_ratings.items():
                objective = key[candidate].get("metrics", {}).get(metric)
                if questions.get(name) and isinstance(objective, (int, float)):
                    points.append((objective, statistics.fmean(questions[name])))
            if len(points) < 3:
                continue
            x, y = map(list, zip(*points))
            if len(set(x)) > 1 and len(set(y)) > 1:
                correlations[f"{name}:{metric}"] = {"candidates": len(points),
                    "pearson": statistics.correlation(x, y),
                    "spearman": statistics.correlation(ranks(x), ranks(y))}
    payload = {"schema_version": 1, "rating_rows": len(ratings),
               "listeners": len({row["listener_id"] for row in ratings}),
               "summary": summary, "ratings": ratings, "metric_correlations": correlations,
               "limitations": ["Only supplied ratings are summarized; blank templates are not judgments.",
                               "Per-question counts differ when listeners leave items unrated.",
                               "Means of ordinal ratings are descriptive; repeated listeners and works are not independent trials.",
                               "Correlations require at least three rated candidates and nonconstant values; they are exploratory, dependent, and subject to multiple comparisons.",
                               "No statistical significance or quality conclusion is inferred from a small listening study."]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--report", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("outputs/human_evaluation"))
    prepare.add_argument("--seed", type=int, default=0)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--key", type=Path, required=True)
    aggregate.add_argument("--ratings", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, default=Path("outputs/human_evaluation/ratings_summary.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_blind_evaluation(args.report, args.output_dir, seed=args.seed)
        else:
            result = aggregate_ratings(args.key, args.ratings, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result if args.command == "prepare" else {k: result[k] for k in ("rating_rows", "listeners", "summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
