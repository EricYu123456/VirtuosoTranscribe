"""Weak pair mining from symbolic catalog metadata; candidates require review."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass, asdict, field, replace
import json
import math
from pathlib import Path
import re
import unicodedata


def normalize_title(title: str) -> str:
    title = unicodedata.normalize("NFKD", title).casefold()
    title = "".join(character for character in title if not unicodedata.combining(character))
    title = re.sub(r"\([^)]*(?:piano|arrang|transcri|voice|violin|cello)[^)]*\)", " ", title)
    title = re.sub(r"\b(?:for|solo|piano|voice|violin|cello|arrangement|arranged|transcription|arr|version)\b", " ", title)
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def catalog_role(row: dict) -> str | None:
    programs = [int(x) for x in re.findall(r"\d+", str(row.get("tracks", "")))]
    if programs:
        if all(program == 0 for program in programs):
            return "target"
        if 0 in programs and sum(program != 0 for program in programs) == 1:
            return "source"
    return None


def metadata_license_eligible(row: dict) -> bool:
    def is_true(value):
        return value is True or str(value).lower() == "true"
    def is_false(value):
        return value is False or str(value).lower() == "false"
    return (str(row.get("license", "")).lower() in {"cc0", "cc-zero", "publicdomain"}
            and is_false(row.get("license_conflict"))
            and is_true(row.get("subset:all_valid", False))
            and is_false(row.get("has_paywall"))
            and is_false(row.get("is_official")))


@dataclass(frozen=True)
class PairCandidate:
    source_path: str
    target_path: str
    normalized_title: str
    confidence: float
    source_metadata: str
    target_metadata: str
    review_status: str = "NEEDS_REVIEW"
    metadata_evidence: dict = field(default_factory=dict)
    score_review: dict = field(default_factory=dict)


def _text(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "")).casefold()
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9#]+", " ", normalized).strip()


def _composer(row: dict) -> str:
    # Catalogs commonly alternate family/given-name order.
    value = _text(row.get("composer_name") or row.get("composer"))
    if value in {"none", "nan", "unknown", "anonymous", "traditional", "not available"}:
        return ""
    return " ".join(sorted(value.split()))


def _ratio(left: object, right: object) -> float | None:
    try:
        left, right = float(left), float(right)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(left) or not math.isfinite(right) or left <= 0 or right <= 0:
        return None
    return min(left, right) / max(left, right)


def _value(row: dict, names: tuple[str, ...]):
    return next((row[name] for name in names if row.get(name) not in (None, "")), None)


def _metadata_evidence(source: dict, target: dict) -> dict:
    left_composer, right_composer = _composer(source), _composer(target)
    duration = _ratio(_value(source, ("song_length.beats", "duration_beats", "duration")),
                      _value(target, ("song_length.beats", "duration_beats", "duration")))
    structure = _ratio(_value(source, ("song_length.bars", "measure_count", "n_measures")),
                       _value(target, ("song_length.bars", "measure_count", "n_measures")))
    evidence = {"composer_agreement": left_composer == right_composer if left_composer and right_composer else None,
                "duration_similarity": duration, "measure_count_similarity": structure,
                "instrumentation": "catalog solo+piano to piano-only", "title_match": "exact normalized title"}
    for name, aliases in (("key_agreement", ("key", "key_signature", "tonal_center")),
                          ("meter_agreement", ("meter", "time_signature"))):
        left, right = _value(source, aliases), _value(target, aliases)
        evidence[name] = _text(left) == _text(right) if left is not None and right is not None else None
    return evidence


def mine_pairs(rows: list[dict], *, review_scores: bool = False,
               score_root: str | Path | None = None, min_melody_coverage: float = .75,
               min_alignment_quality: float = .6, min_structural_similarity: float = .65,
               onset_tolerance: float = .125, include_rejected: bool = False,
               max_candidates: int | None = None, max_score_events: int = 20000,
               max_alignment_cells: int = 2000000) -> list[PairCandidate]:
    """Propose pairs, optionally reviewing existing local score files.

    Metadata mode never parses scores or downloads files. Score review is an
    automatic filtering aid, not musical or legal approval; every surviving
    candidate still requires an individual human review. Rejections and missing
    local files can be retained explicitly for an audit report.
    """
    for name, value in (("min_melody_coverage", min_melody_coverage),
                        ("min_alignment_quality", min_alignment_quality),
                        ("min_structural_similarity", min_structural_similarity)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in 0..1")
    if not isinstance(onset_tolerance, (int, float)) or isinstance(onset_tolerance, bool) or not math.isfinite(onset_tolerance) or onset_tolerance < 0:
        raise ValueError("onset_tolerance must be finite and non-negative")
    for name, value in (("max_candidates", max_candidates), ("max_score_events", max_score_events),
                        ("max_alignment_cells", max_alignment_cells)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    groups = defaultdict(lambda: {"source": [], "target": []})
    for row in rows:
        if not metadata_license_eligible(row):
            continue
        role = catalog_role(row)
        name = normalize_title(str(row.get("song_name") or row.get("title", "")))
        if role and name:
            groups[name][role].append(row)
    pairs = []
    for name, roles in groups.items():
        for source in roles["source"]:
            for target in roles["target"]:
                # Duration and composer support narrow a title-based proposal;
                # only score/note alignment can establish an actual musical pair.
                evidence = _metadata_evidence(source, target)
                if evidence["composer_agreement"] is False:
                    continue
                source_path, target_path = str(source.get("mxl") or ""), str(target.get("mxl") or "")
                if not source_path or not target_path or source_path == target_path:
                    continue
                confidence = .5 + .2 * bool(evidence["composer_agreement"]) + .3 * (evidence["duration_similarity"] or 0)
                # Mismatched keys can reflect a legitimate transposition; treat
                # them as weak evidence rather than an automatic rejection.
                if evidence["key_agreement"] is False:
                    confidence *= .9
                if evidence["meter_agreement"] is False:
                    confidence *= .8
                if evidence["measure_count_similarity"] is not None:
                    confidence *= .8 + .2 * evidence["measure_count_similarity"]
                candidate = PairCandidate(source_path, target_path, name, confidence,
                                          str(source.get("metadata", "")), str(target.get("metadata", "")),
                                          metadata_evidence=evidence)
                pairs.append((candidate, source))
    pairs.sort(key=lambda item: (-item[0].confidence, item[0].normalized_title, item[0].target_path))
    if max_candidates is not None:
        pairs = pairs[:max_candidates]
    if not review_scores:
        return [candidate for candidate, _ in pairs]
    thresholds = {"min_melody_coverage": min_melody_coverage,
                  "min_alignment_quality": min_alignment_quality,
                  "min_structural_similarity": min_structural_similarity,
                  "onset_tolerance_quarter_notes": onset_tolerance,
                  "max_score_events": max_score_events, "max_alignment_cells": max_alignment_cells}
    parsed_cache: dict = {}
    result = []
    root = Path(score_root).resolve() if score_root is not None else Path.cwd()
    for candidate, source_row in pairs:
        review = _review_local_pair(candidate, source_row, root, parsed_cache, thresholds)
        if review["status"] != "PASSED_THRESHOLDS" and not include_rejected:
            continue
        if review["status"] == "PASSED_THRESHOLDS":
            confidence = (.2 * candidate.confidence + .4 * review["melody_coverage_octave_tolerant"]
                          + .2 * review["alignment_quality"] + .2 * review["structural_similarity"])
        else:
            confidence = 0.
        result.append(replace(candidate, confidence=confidence, score_review=review))
    return sorted(result, key=lambda pair: (-pair.confidence, pair.normalized_title, pair.target_path))


def _review_local_pair(candidate: PairCandidate, source_row: dict, root: Path,
                       parsed_cache: dict, thresholds: dict) -> dict:
    review = {"status": "UNAVAILABLE", "thresholds": dict(thresholds),
              "method": "local_score_role_validation_and_measure_onset_group_alignment",
              "legal_review": "NEEDS_REVIEW", "musical_review": "NEEDS_REVIEW",
              "limitations": ["Automatic coverage cannot infer the target melody voice.",
                              "Passing numerical thresholds does not approve provenance or artistic pair quality.",
                              "No key transposition search or formal reordering is performed."]}
    paths = []
    for label, value in (("source", candidate.source_path), ("target", candidate.target_path)):
        if re.match(r"^[a-z]+://", value, flags=re.I):
            review["reason"] = f"{label} is a remote URL; score review never downloads"
            return review
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        review[f"resolved_{label}_path"] = str(path)
        if not path.is_file():
            review["reason"] = f"Local {label} score does not exist"
            return review
        paths.append(path)
    from src.analysis.melody_alignment import align_solo_to_piano
    from src.data.identify_roles import is_piano
    from src.data.normalize_score import normalize_source
    from src.data.parse_musicxml import parse_score
    try:
        scores = []
        for path in paths:
            if path not in parsed_cache:
                try:
                    parsed_cache[path] = parse_score(path)
                except (OSError, ValueError) as exc:
                    parsed_cache[path] = exc
            value = parsed_cache[path]
            if isinstance(value, Exception):
                raise value
            if len(value.notes) > thresholds["max_score_events"]:
                raise ValueError(f"Score has {len(value.notes)} events, above the configured review limit; segment it for review")
            scores.append(value)
        source = normalize_source(scores[0], solo_part=source_row.get("solo_part"),
                                  accompaniment_parts=source_row.get("accompaniment_parts"))
        target = scores[1]
        if not target.notes or not target.parts or not all(is_piano(part) for part in target.parts):
            review.update(status="REJECTED_INSTRUMENTATION", reason="Actual target score is not non-empty piano-only music")
            return review
        cells = ((len({note.onset for note in source.solo}) + 1)
                 * (len({note.onset for note in target.notes}) + 1))
        if cells > thresholds["max_alignment_cells"]:
            review.update(status="REVIEW_LIMIT_EXCEEDED", alignment_cells=cells,
                          reason="Onset-group alignment exceeds the configured cell budget; segment the work for review")
            return review
        alignment = align_solo_to_piano(source.solo, target, source_score=source.score,
                                        onset_tolerance=thresholds["onset_tolerance_quarter_notes"])
        coverage = len(alignment.matches) / len(source.solo)
        duration_similarity = _ratio(source.duration, target.duration) or 0.
        measure_similarity = _ratio(len(source.score.measures), len(target.measures)) or 0.
        source_measures = {measure.number: measure for measure in source.score.measures}
        target_measures = {measure.number: measure for measure in target.measures}
        meter_agreement = (sum(source_measures[match.source_measure].time_signature == target_measures[match.target_measure].time_signature
                               for match in alignment.measure_matches) / len(alignment.measure_matches)
                           if alignment.measure_matches else 0.)
        structure = .4 * duration_similarity + .4 * measure_similarity + .2 * meter_agreement
        keys_left = [event.value for event in source.score.key_signatures]
        keys_right = [event.value for event in target.key_signatures]
        review.update(melody_coverage_octave_tolerant=coverage,
                      melody_coverage_exact=sum(match.source_pitch == match.target_pitch for match in alignment.matches) / len(source.solo),
                      alignment_quality=alignment.quality, matched_melody_notes=len(alignment.matches),
                      source_solo_note_count=len(source.solo), target_note_count=len(target.notes),
                      duration_similarity=duration_similarity, measure_count_similarity=measure_similarity,
                      meter_agreement=meter_agreement,
                      key_signature_agreement=keys_left == keys_right if keys_left and keys_right else None,
                      structural_similarity=structure,
                      source_solo_part_ids=list(source.solo_part_ids),
                      alignment=alignment.to_dict())
        failures = []
        if coverage < thresholds["min_melody_coverage"]:
            failures.append("melody_coverage_octave_tolerant")
        if alignment.quality < thresholds["min_alignment_quality"]:
            failures.append("alignment_quality")
        if structure < thresholds["min_structural_similarity"]:
            failures.append("structural_similarity")
        review.update(status="REJECTED_THRESHOLDS" if failures else "PASSED_THRESHOLDS",
                      failed_thresholds=failures)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        review.update(status="PARSE_OR_ROLE_ERROR", reason=str(exc))
    return review


def load_catalog(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload.get("candidates", [])


def save_candidates(candidates: list[PairCandidate], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(candidate) for candidate in candidates], indent=2,
                               ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path
