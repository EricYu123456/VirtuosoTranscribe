# Implementation status

Status: 2026-10-05. The V1 M0–M10 engineering pipeline in [the specification](../VirtuosoTranscribe_PROJECT.md) is implemented and exercised. This is a research prototype; the external corpus scale and human study targets are reported separately from software completion.

## Milestones

| Milestone | Implemented and checked | Boundary |
| --- | --- | --- |
| M0 Repository | Python package, locked CPU dependencies, CLI entry points, wheel with demo assets, tests, configs, examples. | Checkpoints/data live in ignored workspace artifact directories. |
| M1 Round trip | MusicXML/MXL/MIDI, concert pitch, resolved ties, pickup timing, meter/key/tempo/dynamics/tuplets/slurs, polyphonic MusicXML and MIDI export. | Canonical musical events are tested; arbitrary engraving and every notation feature are not reproduced. |
| M2 Roles/tokens | Explicit SOLO/ACC streams; part selectors; archival bundles; exact-grid event tokens and detokenization. | Ambiguous roles need selectors; bare tokens do not contain all notation. |
| M3 Alignment | Measure DP, beat mapping, one-to-one octave-equivalent melody matching, JSON and interactive SVG/HTML review. | Monotonic form; same-pitch accompaniment can be mistaken for melody. |
| M4 Metrics | Melody, harmonic, structural, texture/register, technique proxies and sustained-note playability. | Phrase/cadence/artistic voice identity and pianist approval are not inferred. |
| M5 Baselines | Literal reduction and rule-based voicing, melody/bass protection, octave doubling, reinforcement, chord thickness, inner voices, figuration and density. | Protected unplayable solo remains with warnings; unsupported compositional freedoms fail explicitly. |
| M6 Learned seq2seq | Single/dual Transformer, causal/padding masks, AdamW/warmup, validation, gradient clipping/accumulation, seeded resumable weights-only checkpoints. | Tiny checkpoints establish learning, not generalization. |
| M7 Melody constraints | none/soft/hard grammar-constrained autoregression; differentiable aligned melody loss; reports forced events and raw tokens. | Hard mode is exact or ±12 with source onset/duration; unsupported grace notation/context fails. |
| M8 Conditioning | All 13 continuous controls projected into encoder memory and decoder; trained octave-doubling control variants. | Demonstrated learned response is octave doubling. Other artistic techniques require annotated training examples. |
| M9 Reranking | Distinct sampled/beam N-best candidates; melody/harmony/playability/technique scoring; optional supplied MusicBERT content adapter. | Heuristic weights; genuine pretrained MusicBERT has not been run here. |
| M10 Demo | Local upload/presets/sliders, checkpoint selection, three melody modes, source/target audio, alignment roll, metrics, XML/MIDI/WAV download. | Synthetic audio by default; optional FluidSynth needs a user-supplied SoundFont. |

## Actual learned-model evidence

All results below are original synthetic auxiliary-data sanity checks, with 10 train works, one validation work and one reserved test work. Control variants stay with their parent work.

| Model | Train / validation segments | Final CE | Raw autoregressive complete-target matches |
| --- | ---: | ---: | ---: |
| Vanilla single encoder | 10 / 1 | 0.000799 | 10 / 10 |
| Unconditioned dual encoder | 10 / 1 | 0.000998 | 10 / 10 |
| Conditioned single encoder | 20 / 2 | 0.001000 | 20 / 20 |

Raw checks use no target prefix, lookup, grammar repair or melody constraint. The conditioned model reproduces 24-note/plain and 40-note/octave-doubled targets from the same source by changing controls. A correctly learned hard-decoded example preserves all 16 solo attacks with zero forced pitch/duration changes and zero post-generation projection repairs.

Held-out pitch-token accuracy is only 45.83% / 50.00% / 43.75%; validation CE is 0.6297 / 0.7252 / 0.5715. Overall token accuracy is dominated by syntax. These figures do not demonstrate artistic generalization or model superiority.

Machine-readable results, dataset SHA256, RNG/resume review and exact initial/refinement commands are in `outputs/checkpoints/training_summary.json`. Raw generated tokens are in each checkpoint's `overfit_generation.json` and `validation_generation.json`.

## Experiment and human-evaluation pipeline

The genuine Schubert/August Horn pair is also trained: matched single, dual and conditioned models use the same ten fully aligned bars (nine distinct source inputs), one work, train only. Final CE is 0.000997 / 0.001000 / 0.000997, and all three raw autoregressive checks reproduce 10/10 complete targets including EOS. Exact commands and provenance are in `outputs/checkpoints/real_training_summary.json`. These results establish learning on real historical notation, with no held-out artistic claim.

`configs/evaluation/full.yaml` runs 15 methods covering the five main experiment families: baselines, role separation, melody constraints, technique control, and N-best reranking, plus reference targets. The initial three-fixture run produced 45 successful rows, zero omitted methods and zero failures, with frozen checkpoint SHA256 provenance and explicit diagnostic labels.

The fixtures are related variants, so this run is an integration check. The control results are not universal: the mean rule-based octave rate changed 0.231 → 1.000; the conditioned checkpoint changed 0.000 → 0.077 on these out-of-distribution fixtures. Never infer trained artistic technique quality from perfect hard-constraint coverage.

The real historical run also produced 45 successful rows, zero omissions and zero failures at `outputs/experiments/real/index.html`. The source and original Horn target excerpts (bars 5/6/13) are rendered at `outputs/real_examples/`; the real anonymous listening package is `outputs/human_evaluation_real/participant/`. Vanilla and dual reproduce their three in-sample targets exactly. Conditioned generation with a Romantic request changes texture; its runtime controls differ from the native training annotations, so exact reconstruction is not claimed for every preset.

The listening package anonymizes candidate labels and keeps method identities in a separate private key. It includes source/reference listening material, piano rolls, scores and seven 1–5 questions. The aggregator accepts only actual supplied rating CSVs and can compute exploratory metric/rating correlations. Current human observations: **0 listeners, 0 ratings**.

## Data and reproducibility

`prepare_dataset` checks both legal-status declarations before parsing, prohibits work-level split leakage, retains clipped-note provenance, and passes explicit control/data-kind annotations into segments. Transposition augmentation updates key signatures and respects pitch bounds. Weak pair mining uses title/composer/instrumentation metadata and optional local score-alignment review; candidates never become CHECKED automatically.

Real source/transcription acquisition and license evidence are recorded in [the dataset card](dataset_card.md). Synthetic and real data must remain identified separately. The Stage A 20–50-work recommendation and Stage B hundreds/thousands recommendation are corpus-growth targets; implemented collection/preparation/mining tools do not themselves meet those counts.

## Verification and remaining research acceptance

Tests cover musical event round trips, pickups/ties/tuplets/transposition, compound/additive meter, roles, exact-grid timing, alignment, sustained-note capacity/collisions, controls, differentiable gradients, causal masks, learned reconstruction, exact RNG resume, constrained decoding, genuine adapter contract, audio duration/clipping, HTTP upload/download and blind-package integrity. Run `pytest -q`, `ruff check .`, and `pip check`; the wheel also builds with its HTML asset included.

Final validation: 179 tests pass, Ruff passes, pip reports no broken requirements, and `uv lock --check` passes. The three PyTorch warnings concern deprecated TorchScript APIs used only by the optional adapter-contract test. The wheel includes its demo HTML asset.

The browser workflow was exercised end to end with a rule-based profile and a genuine trained historical checkpoint: upload → generation → metrics/alignment → playable WAV and score downloads. Screenshots are `outputs/demo_verified/screenshot.jpg` and `outputs/demo_verified/real_model.jpg`. The running loopback demo exposes all six checkpoints.

External research acceptance remains open: larger individually checked artistic pairs, reliable work-disjoint generalization, pianist/listener ratings, calibrated playability and genuine pretrained MusicBERT evaluation. V2/V3 stretch goals are explicitly outside the V1 implementation scope.

