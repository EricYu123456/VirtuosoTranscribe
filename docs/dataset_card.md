# Dataset card

Status: 2026-10-05. Real historical data, original synthetic auxiliary data, and related pipeline fixtures are distinct datasets. Source/target rights are checked separately, and every derivative retains its parent's work ID and split.

## Inventory

| Dataset | Works / pairs / clips | Purpose and split |
| --- | --- | --- |
| Bundled `examples/` | 1 original study / 3 variants | Cello, violin and voice plus piano; all train. Related variants cannot be independent validation observations. |
| `make_training_data` | 12 original motifs / 12 plain pairs or 24 controlled variants | 10 train works, 1 validation work, 1 reserved test work. Plain/doubled targets are synthetic auxiliary arrangements. |
| Curated historical seed | 1 work / 1 complete pair / 59 one-bar aligned clips | Schubert D957/4, voice and piano → August Horn solo-piano transcription. All train, no real held-out works. |
| Real overfit subset | 10 selected clips / 9 distinct source inputs | Exact melody alignment, at least two solo notes, no grace tokens, bounded context. Repeated authentic bars have identical targets; there is no contradictory supervision. |
| Historical listening excerpts | Source bars 5, 6, 13, with their actual mapped target bars | Derivatives of the same real work, all train; not three new works. |

The complete historical source has 1,114 notes, including 120 solo attacks, in 59 measures. The target has 1,471 notes in 82 measures, including 14 grace notes. Both are 3/4 with a one-flat signature. Full monotonic alignment matches 119/120 solo attacks (99.17%); one source attack remains unmatched. Different repeat/form treatment prevents assuming identical full-score timelines.

## Historical provenance and permissions

The source is OpenScore Lieder `lc5004835.mxl`, pinned to commit `38c5db510224d9facdc4b08d741fc788cfb58ea8`. Its XML says OpenScore (CC0), and the repository supplies the [CC0 license](https://raw.githubusercontent.com/OpenScore/Lieder/38c5db510224d9facdc4b08d741fc788cfb58ea8/LICENSE.txt).

The target is individually selected PDMX score 6212007. Archived original MuseScore metadata identifies an exact replica of August Horn's transcription and a [CC0 copyright waiver](https://creativecommons.org/publicdomain/zero/1.0/); its XML credits identify Horn. The metadata's D889 work label was corrected to D957/4 using the actual melody, key, meter and 119/120 matches. The source page is [the Horn score](https://musescore.com/user/30059877/scores/6212007).

[PDMX v9](https://zenodo.org/records/15571083) warns about internal/public license discrepancies and recommends the no-license-conflict subset. This selected record is CC-zero, no conflict, all-valid, nonofficial and nonpaywalled. Both downloaded archive MD5 values match the upstream record; source, target and metadata SHA256 are recorded.

The checked manifest is `data/metadata/curated_pairs.json`; the compact evidence is `data/metadata/license_audit.json`. Raw primary snapshots remain under `data/raw/pdmx_seed`; the public-field inspection report is `outputs/corpus_review/target_provenance.json`. CHECKED records reviewed evidence; the manifest validator does not independently establish legal rights.

## Candidate search and exclusions

All 254,077 PDMX rows were inspected, retaining 511 Schubert/Liszt/title candidates. The candidate cache is `data/metadata/pdmx_candidates.json`; its initial eligibility flag is a snapshot from that scan, and current eligibility is recomputed by the mining code. Seven targets and five CC0 source scores were acquired for review.

Actual score inspection rejected ambiguous title collisions, monophonic reels, accompaniment-only arpeggios and lead-sheet/folk settings without verified full source/transcription correspondence. The Musetrainer Liszt copy lacks independently verified digital permissions and is excluded. These files are candidates, not CHECKED pairs.

Lead-sheet chord symbols revealed and fixed an import issue: unsounded `ChordSymbol` annotations are excluded from pitched events, with a warning. Overfull measures retain actual timing with an explicit warning. These fixes do not turn unrelated scores into accepted transcriptions.

Weak mining supports cached local instrument/measure/note/key/meter review and explicit thresholds; candidates remain NEEDS_REVIEW. Corpus Stage A's approximately 20–50 works and Stage B's hundreds/thousands are **not achieved** by this one-work seed. Stage C tooling is implemented; broad pair-quality and rights review remain work to do.

## Reproduction

```powershell
# Uses the already reviewed hashes/manifest; an uncached PDMX download is ~2 GB
.\.venv\Scripts\python -m scripts.fetch_curated_seed
.\.venv\Scripts\python -m scripts.prepare_dataset --manifest data/metadata/curated_pairs.json --output data/processed/curated --segment-measures 1
.\.venv\Scripts\python -m scripts.select_overfit_segments
.\.venv\Scripts\python -m scripts.train_model --config configs/train/real_overfit.yaml
.\.venv\Scripts\python -m scripts.train_model --config configs/train/real_dual.yaml
.\.venv\Scripts\python -m scripts.train_model --config configs/train/real_conditioned.yaml
.\.venv\Scripts\python -m scripts.export_real_examples --manifest data/metadata/curated_pairs.json --processed data/processed/curated/dataset.json --output-dir outputs/real_examples --source-measure 5 --source-measure 6 --source-measure 13
```

Processed artifacts retain role streams, archival notation bundles, parent identifiers, explicit data kinds/control annotations and clipped-note provenance. Transposition changes both score pitches and key signatures without moving data between splits. Default grid is 24 subdivisions per quarter; unrepresentable rhythms fail explicitly.

## Interpretation limits

The synthetic and real models' perfect raw train reconstruction validates representation and optimization. One real work cannot demonstrate held-out artistic quality, learned arranger style or comparative superiority. Synthetic validation includes only one work and has about 44–50% pitch accuracy. All real listening comparisons are labeled in-sample diagnostics.

Continuous real training controls are target-feature annotations, an intentional supervised conditioning input. Runtime generation receives only the supplied controls and source; it does not retrieve a reference target. Only synthetic octave doubling has a controlled same-source learning demonstration.

Listening WAVs use the same additive piano-like renderer, not a pianist recording. The anonymous questionnaires and aggregator are implemented; actual observations are **0 listeners / 0 ratings**. No preference, expression or physical-playability conclusions have been inferred.
