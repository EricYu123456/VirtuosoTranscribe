# VirtuosoTranscribe

## Expressive Piano-Solo Transcription from Solo-and-Accompaniment Scores

**Project type:** Symbolic Music / MIR / Generative Modeling  
**Primary domain:** Classical music, especially voice, violin, and cello works with accompaniment  
**Primary output:** Piano-solo MusicXML + MIDI  
**Status:** Implementation specification / MVP plan

---

## 1. Project Summary

**VirtuosoTranscribe** converts a score containing:

- a clearly identified **solo line** (voice, violin, cello, or another melodic instrument), and
- an existing **accompaniment** (initially piano)

into an **idiomatic piano-solo transcription**.

The goal is **not** to merely copy the solo melody into the right hand and preserve the accompaniment underneath it. The system should reorganize the complete musical material into a convincing piano texture while preserving the musical identity of the original.

Historical piano transcription provides the musical motivation. In song and instrumental transcriptions, arrangers such as **Franz Liszt** and **Sergei Rachmaninoff** frequently preserve the melodic identity of the source while changing register, texture, voicing, doubling, figuration, bass weight, and pianistic density. Jazz pianists perform related transformations when turning a melody-plus-accompaniment source into a complete solo-piano performance.

The project therefore models piano arrangement as a structured transformation:

```text
SOURCE SCORE
solo line + accompaniment
        |
        v
CONTENT / STRUCTURE ENCODING
melody, harmony, rhythm, phrase, bass, texture
        |
        +---- arrangement controls
        |
        v
PIANISTIC TRANSFORMATION MODEL
        |
        v
PIANO SOLO
MusicXML + MIDI
```

The first version should focus on **faithful but expressive transcription**, not free fantasy or operatic paraphrase.

---

## 2. Core Research Question

> Can a symbolic music model transform a solo-and-accompaniment score into an expressive and playable piano-solo transcription while explicitly preserving the identity of the original solo melody?

Secondary questions:

1. Does explicitly separating **solo** and **accompaniment** source streams improve transcription quality?
2. Does **melody-constrained decoding** improve melodic preservation without making the output rigid?
3. Can interpretable **arrangement-technique controls** produce meaningful differences in texture?
4. Is technique-conditioned generation more useful than a single opaque style label such as `LISZT` or `JAZZ`?
5. Can a model learn when to preserve accompaniment literally and when to rewrite it pianistically?
6. Which objective symbolic metrics correlate best with human judgments of:
   - melody preservation,
   - pianistic quality,
   - playability,
   - expressiveness?

---

## 3. Scope

### 3.1 MVP Input

The MVP accepts symbolic scores in **MusicXML** or **MIDI** with two logical musical roles:

```text
SOLO
- voice
- violin
- cello

ACCOMPANIMENT
- piano
```

Preferred initial source type:

```text
solo instrument / voice + piano accompaniment
```

This is intentionally easier than:

```text
solo + orchestra
```

because the harmonic and structural content is already close to the target instrument.

### 3.2 MVP Output

The model produces:

```text
piano solo
```

with:

- MusicXML
- MIDI
- optional rendered audio
- optional piano-roll visualization

The target should be performable by one pianist.

### 3.3 Explicit Non-Goals for V1

Do **not** attempt these in the first implementation:

- raw audio -> piano arrangement
- source separation
- automatic audio transcription
- full orchestral reduction
- operatic fantasy / paraphrase generation
- free recomposition of form
- automatic fingering
- pedal generation
- performer-specific expressive timing
- exact imitation of one named composer

These can become later extensions.

---

## 4. Musical Formulation

A source score contains two kinds of information.

### 4.1 Content to Preserve

The system should strongly preserve:

- primary melody
- phrase boundaries
- meter
- key / tonal center
- harmonic trajectory
- important bass motion
- cadential locations
- important rhythmic motives
- large-scale form

### 4.2 Material the System May Transform

The system may change:

- register
- voicing
- hand distribution
- texture
- note density
- octave placement
- octave doubling
- chord thickness
- accompaniment figuration
- inner voices
- bass reinforcement
- arpeggiation
- tremolo
- repeated-note writing
- countermelody
- ornamentation
- transition filling
- local harmonic color

### 4.3 Material That Requires Explicit Permission

These should be weak or disabled in V1 unless controlled:

- melody note substitution
- phrase deletion
- phrase insertion
- strong reharmonization
- metric displacement
- formal reordering
- cadenza insertion

---

## 5. Arrangement Technique Vocabulary

The project should represent arrangement style through **interpretable techniques**, not only composer names.

Recommended control dimensions:

| Control | Meaning |
|---|---|
| `melody_preservation` | strictness of original melodic pitch/rhythm preservation |
| `octave_doubling` | frequency of melodic octave reinforcement |
| `register_expansion` | degree of vertical expansion across the keyboard |
| `chord_thickening` | tendency to add chord tones / richer voicing |
| `bass_reinforcement` | octave basses, tenths, larger low-register support |
| `inner_voice_density` | amount of added middle-register material |
| `figuration_density` | arpeggios, broken chords, running figures |
| `countermelody` | amount of secondary melodic motion |
| `ornamentation` | turns, passing notes, fills, decorative figures |
| `harmonic_freedom` | freedom to alter or enrich source harmony |
| `rhythmic_freedom` | freedom to displace or reinterpret rhythm |
| `virtuosity` | aggregate technical density / difficulty |
| `texture_density` | total sounding-note density |

Possible presets can later map onto these continuous controls:

```text
FAITHFUL
ROMANTIC
VIRTUOSIC
JAZZ_BALLAD
CONCERT_TRANSCRIPTION
```

Example:

```yaml
preset: ROMANTIC
melody_preservation: 0.98
octave_doubling: 0.55
register_expansion: 0.75
chord_thickening: 0.65
bass_reinforcement: 0.60
inner_voice_density: 0.55
figuration_density: 0.65
countermelody: 0.30
ornamentation: 0.40
harmonic_freedom: 0.20
rhythmic_freedom: 0.15
virtuosity: 0.65
```

Composer names such as Liszt or Rachmaninoff may be used later as **analysis labels or presets derived from observed technique distributions**, but the model should not initially assume that a single token captures an arranger's style.

---

## 6. Recommended Data Strategy

Data is the central challenge of this project.

A large generic MIDI corpus is less useful than a smaller set of **high-quality paired source/transcription examples**.

### 6.1 Ideal Training Pair

```text
SOURCE:
original work
solo + accompaniment

TARGET:
published / curated piano-solo transcription
of the same work
```

Examples may include:

- song + piano -> historical piano-solo transcription
- violin + piano -> piano-solo transcription
- cello + piano -> piano-solo transcription
- other melodic instrument + accompaniment -> piano solo

All training material must be individually checked for legal/public-domain status before redistribution or model release.

### 6.2 Data Sources

Potential sources include:

1. **PDMX**
   - large public-domain symbolic score dataset
   - MusicXML-oriented
   - useful for mining arrangements, instrumentation, titles, and alternate versions

2. Public-domain score repositories
   - useful for manually curated source/transcription pairs

3. Manually entered / corrected examples
   - acceptable for a small research-quality validation set

4. Synthetic pairs
   - useful only as auxiliary pretraining data
   - should not replace real artistic transcriptions

### 6.3 Data Collection Stages

#### Stage A — Curated Seed Corpus

Start with approximately:

```text
20-50 works
```

for pipeline development.

For each work:

```text
source.musicxml
target.musicxml
metadata.json
```

This is enough to build parsing, alignment, visualization, metrics, and deterministic baselines.

#### Stage B — Medium Paired Corpus

Target:

```text
hundreds of paired works
or
thousands of aligned segments
```

Segment long works into phrase- or measure-level examples.

#### Stage C — Large Weakly Paired Corpus

Mine candidate alternate arrangements from large symbolic datasets using:

- normalized title
- composer
- instrumentation
- key
- melody similarity
- structural similarity

Then manually or automatically filter pair quality.

---

## 7. Canonical Data Schema

Store every example in a normalized intermediate representation.

Example:

```json
{
  "pair_id": "example_0001",
  "work_id": "normalized_work_name",
  "composer": "Source Composer",
  "arranger": "Target Arranger",
  "source": {
    "path": "data/raw/source/example.musicxml",
    "solo_instrument": "cello",
    "accompaniment": ["piano"],
    "key": "D minor",
    "meter": "4/4"
  },
  "target": {
    "path": "data/raw/target/example.musicxml",
    "instrument": "piano"
  },
  "license": {
    "source_status": "CHECKED",
    "target_status": "CHECKED"
  },
  "alignment": {
    "method": "measure_beat_note",
    "quality": 0.94
  },
  "split": "train"
}
```

Processed segment:

```json
{
  "segment_id": "example_0001_m001_m008",
  "pair_id": "example_0001",
  "start_measure": 1,
  "end_measure": 8,
  "source_solo_tokens": [],
  "source_acc_tokens": [],
  "target_piano_tokens": [],
  "technique_features": {}
}
```

---

## 8. Score Preprocessing

Recommended initial library:

- `music21`

Useful supporting libraries:

- `pretty_midi`
- `mido`
- `numpy`
- `scipy`
- `pandas`

Optional later:

- `miditok`
- `muspy`
- PDMX's own tooling

### 8.1 Normalization

For every source/target score:

1. Parse MusicXML.
2. Resolve tied notes.
3. Normalize tempo representation.
4. Normalize measure numbering.
5. Preserve:
   - time signatures,
   - key signatures,
   - dynamics when reliable,
   - articulations when reliable,
   - slurs if available.
6. Quantize symbolic time to a controlled grid.
7. Keep tuplets explicitly where possible.
8. Convert transposing instruments to sounding or concert pitch consistently.
9. Identify solo part.
10. Identify accompaniment parts.
11. Extract measure and beat indices.
12. Export a normalized representation for debugging.

Do **not** discard MusicXML too early. MIDI alone loses notation information that may later become useful.

---

## 9. Alignment

Alignment between source and target is required for:

- training,
- melody-preservation metrics,
- technique extraction,
- interpretability.

Use a hierarchical strategy.

### 9.1 Level 1 — Measure Alignment

Align using:

- measure count
- time signatures
- phrase boundaries
- key changes
- approximate duration

### 9.2 Level 2 — Beat Alignment

Within aligned measures, use:

- beat position
- harmonic rhythm
- source solo onsets
- bass onsets

### 9.3 Level 3 — Melody Note Alignment

For the source solo and target piano:

Use dynamic programming / sequence alignment with a cost based on:

```text
pitch-class difference
octave-tolerant pitch difference
onset difference
duration difference
local melodic interval similarity
```

Important:

A melody note moved by one octave should normally count as preserved.

Therefore evaluate at least two melody matching modes:

```text
strict_pitch
octave_equivalent_pitch_class
```

---

## 10. Technique Extraction

Before training a generative model, build a **transcription-analysis pipeline**.

This is valuable on its own.

For every aligned source/target pair, estimate:

### 10.1 Melody Preservation

- preserved melody-note ratio
- octave-shift ratio
- melody doubling ratio
- register relocation distribution
- rhythmic alteration ratio

### 10.2 Density

- notes per beat
- notes per measure
- average simultaneous notes
- maximum simultaneous notes

### 10.3 Register

- lowest pitch
- highest pitch
- pitch span
- target expansion relative to source
- bass-register extension

### 10.4 Harmonic Texture

- average chord size
- vertical pitch-class count
- added non-melody tones
- source-to-target harmonic similarity

### 10.5 Pianistic Transformation

Heuristically detect:

- octave doubling
- parallel thirds/sixths
- arpeggiation
- tremolo
- repeated notes
- chordal melody
- inner-voice filling
- bass octaves
- broken-chord accompaniment
- stride-like bass
- large leaps
- register redistribution

The initial classifier can be rule-based.

Later, technique labels can train a learned classifier.

---

## 11. Baselines

Implement baselines **before** a large neural model.

### Baseline 0 — Literal Reduction

```text
solo -> piano top voice
original accompaniment -> retained
collision handling -> simple pitch/register redistribution
```

This should intentionally be musically plain.

Its purpose is to prove that the learned system does more than merge tracks.

### Baseline 1 — Rule-Based Pianistic Reduction

Rules:

- melody placed in upper register
- accompaniment notes colliding with melody are redistributed
- bass preserved
- impossible spans reduced
- optional melody octave doubling
- optional chord thickening
- accompaniment density capped

### Baseline 2 — Vanilla Seq2Seq Transformer

Input:

```text
[SOURCE_SOLO] ...
[SOURCE_ACC] ...
```

Output:

```text
[TARGET_PIANO] ...
```

No explicit technique controls.

This is the main neural baseline.

### Baseline 3 — Technique-Conditioned Transformer

Add continuous or discrete arrangement-control tokens.

---

## 12. Token Representation

Start simple.

A REMI-like or structured event representation is easier for generation than directly reusing MusicBERT's exact input format.

Recommended event fields:

```text
BAR
POSITION
ROLE
PITCH
DURATION
VELOCITY_BIN
VOICE_ID
```

Source role:

```text
ROLE_SOLO
ROLE_ACC
```

Target role:

```text
ROLE_PIANO
```

Optional notation fields:

```text
ARTICULATION
DYNAMIC
TIE
SLUR_ID
```

Do not require them in MVP training.

Example source sequence:

```text
<BAR_12>
<POS_0>
<ROLE_SOLO>
<PITCH_72>
<DUR_8>

<POS_0>
<ROLE_ACC>
<PITCH_48>
<DUR_16>
...
```

### 12.1 Two-Stream Alternative

A stronger architecture can encode:

```text
SOLO STREAM
ACCOMPANIMENT STREAM
```

separately and fuse them using cross-attention.

This should be tested as an ablation against a single concatenated sequence.

---

## 13. MVP Model Architecture

Recommended first learned model:

> **Encoder-decoder Transformer with role-aware source encoding and melody-constrained target decoding**

```text
                    SOURCE
          +-----------------------+
          |                       |
      SOLO TOKENS            ACC TOKENS
          |                       |
       Encoder A               Encoder B
          |                       |
          +----------+------------+
                     |
               cross fusion
                     |
           arrangement controls
                     |
                     v
              Piano Decoder
                     |
                     v
         constrained target tokens
```

Simpler fallback:

```text
concatenated source
        |
Transformer encoder
        |
Transformer decoder
        |
piano tokens
```

Start with the simpler fallback if the dataset is small.

---

## 14. Melody-Constrained Decoding

This is a major feature of the project.

The source melody is known. The model should not be allowed to casually destroy it.

### 14.1 Hard Constraint Mode

For important melody onsets, require a target note that is one of:

```text
exact source pitch
source pitch + 12
source pitch - 12
```

Potentially allow octave doubling.

### 14.2 Soft Constraint Mode

Add a loss:

```text
L_total =
    L_token
  + lambda_melody * L_melody
  + lambda_harmony * L_harmony
  + lambda_playability * L_playability
```

Where `L_melody` penalizes missing aligned melody notes.

### 14.3 Flexible Mode

Allow controlled melodic alteration when:

```text
melody_preservation < threshold
```

This is not required for MVP.

---

## 15. Training Objectives

Start with standard teacher-forced token prediction:

```text
L_token = cross_entropy(target_tokens)
```

Add auxiliary losses only after the baseline works.

Candidate auxiliary objectives:

### Melody Presence Loss

Encourage source melody pitches at aligned target positions.

### Register Loss

Predict expected melody register.

### Technique Prediction Loss

Predict transformation features from generated or hidden representations.

### Harmonic Consistency Loss

Compare pitch-class distributions or chord representations between source and target.

Avoid introducing too many handcrafted losses in the first experiment.

---

## 16. MusicBERT's Role

MusicBERT should be treated primarily as a **symbolic representation/evaluation model**, not necessarily the core decoder.

Potential uses:

### A. Source/Target Embedding Similarity

```text
source content -> MusicBERT -> embedding
target output  -> MusicBERT -> embedding
```

Measure high-level symbolic similarity.

### B. Technique / Style Classifier

Fine-tune embeddings to classify:

- faithful transcription
- romantic transcription
- virtuosic transcription
- jazz-oriented transcription

or individual technique presence.

### C. Candidate Reranking

Generate `N` candidate transcriptions and rank using:

```text
score =
    a * melody_preservation
  + b * content_similarity
  + c * playability
  + d * target_technique_match
```

MusicBERT can contribute to `content_similarity` or learned stylistic scoring.

---

## 17. Playability Modeling

A transcription may be musically plausible but physically unreasonable.

Implement a lightweight rule-based playability scorer.

Features:

- simultaneous hand span
- total simultaneous notes
- impossible sustained-note conflicts
- extreme leaps
- rapid repeated wide jumps
- register collisions
- excessive density
- voice crossing
- unrealistic chord spans

Do not attempt perfect hand-assignment in V1.

Approximate playability is enough for reranking.

Example:

```python
playability_score = (
    1.0
    - span_penalty
    - leap_penalty
    - density_penalty
    - collision_penalty
)
```

Later work can add hand assignment or fingering.

---

## 18. Evaluation

Use both **objective metrics** and **human listening/score evaluation**.

### 18.1 Melody Preservation

Report:

- exact pitch onset F1
- octave-tolerant onset F1
- melodic contour similarity
- rhythm similarity
- melody-note coverage
- melody octave-doubling rate

The octave-tolerant score is especially important for piano transcription.

### 18.2 Harmonic Preservation

Possible metrics:

- beat-level pitch-class overlap
- chord-root agreement
- tonal-distance metric
- bass-motion similarity
- harmonic rhythm similarity

### 18.3 Structural Preservation

- measure count ratio
- phrase-length agreement
- cadence alignment
- key-change agreement

### 18.4 Pianistic Transformation

Report:

- density increase
- register expansion
- added-note ratio
- average polyphony
- octave doubling
- bass reinforcement
- inner-voice density
- figuration density

### 18.5 Playability

Report:

- span violations
- density violations
- leap violations
- estimated difficulty

### 18.6 Human Evaluation

Recommended 1-5 Likert questions:

1. How well is the original melody preserved?
2. How natural is the piano writing?
3. How musically coherent is the arrangement?
4. How expressive is the transcription?
5. How playable does it appear?
6. How much does it sound like an intentional piano arrangement rather than an automatic reduction?
7. Overall preference.

Compare at minimum:

```text
Literal baseline
Rule-based baseline
Vanilla Transformer
Technique-conditioned model
Human transcription target
```

Blind evaluation is strongly preferred.

---

## 19. Main Experiments

### Experiment 1 — Does a Learned Model Beat Literal Reduction?

```text
B0 Literal
vs
B1 Rule-based
vs
B2 Transformer
```

### Experiment 2 — Does Role Separation Help?

```text
single source stream
vs
separate SOLO + ACC encoders
```

### Experiment 3 — Melody Constraint

```text
no constraint
vs
soft constraint
vs
hard constraint
```

Measure musicality and melody preservation.

### Experiment 4 — Technique Conditioning

Hold the source constant and vary:

```text
virtuosity
chord_thickening
octave_doubling
register_expansion
figuration_density
```

Test whether metrics and human perception change in the expected direction.

### Experiment 5 — Candidate Reranking

```text
top-1 decoder output
vs
N-best + rule scorer
vs
N-best + learned/MusicBERT scorer
```

---

## 20. Proposed Repository Structure

```text
virtuoso-transcribe/
|
|-- PROJECT.md
|-- README.md
|-- pyproject.toml
|-- configs/
|   |-- data/
|   |-- model/
|   |-- train/
|   `-- presets/
|
|-- data/
|   |-- raw/
|   |   |-- source/
|   |   `-- target/
|   |-- interim/
|   |-- processed/
|   |-- manifests/
|   `-- metadata/
|
|-- src/
|   |-- data/
|   |   |-- parse_musicxml.py
|   |   |-- normalize_score.py
|   |   |-- identify_roles.py
|   |   |-- segment.py
|   |   |-- align.py
|   |   `-- build_manifest.py
|   |
|   |-- representation/
|   |   |-- events.py
|   |   |-- tokenizer.py
|   |   `-- detokenizer.py
|   |
|   |-- analysis/
|   |   |-- melody_alignment.py
|   |   |-- harmony.py
|   |   |-- texture.py
|   |   |-- technique_features.py
|   |   `-- playability.py
|   |
|   |-- baselines/
|   |   |-- literal_reduction.py
|   |   `-- rule_based.py
|   |
|   |-- models/
|   |   |-- seq2seq.py
|   |   |-- dual_encoder.py
|   |   |-- conditioning.py
|   |   `-- constraints.py
|   |
|   |-- training/
|   |   |-- dataset.py
|   |   |-- collator.py
|   |   |-- losses.py
|   |   `-- trainer.py
|   |
|   |-- evaluation/
|   |   |-- melody_metrics.py
|   |   |-- harmonic_metrics.py
|   |   |-- structural_metrics.py
|   |   |-- playability_metrics.py
|   |   `-- evaluate.py
|   |
|   `-- inference/
|       |-- generate.py
|       |-- rerank.py
|       `-- export.py
|
|-- scripts/
|   |-- prepare_dataset.py
|   |-- train_baseline.py
|   |-- train_model.py
|   |-- generate_examples.py
|   `-- evaluate_all.py
|
|-- tests/
|   |-- test_parser.py
|   |-- test_alignment.py
|   |-- test_tokenizer.py
|   |-- test_metrics.py
|   `-- test_roundtrip.py
|
|-- notebooks/
|   |-- 01_dataset_inspection.ipynb
|   |-- 02_transcription_analysis.ipynb
|   `-- 03_model_results.ipynb
|
|-- outputs/
|   |-- midi/
|   |-- musicxml/
|   |-- audio/
|   |-- figures/
|   `-- evaluation/
|
`-- docs/
    `-- dataset_card.md
```

---

## 21. Suggested CLI

Dataset preparation:

```bash
python -m scripts.prepare_dataset \
    --manifest data/manifests/pairs.csv \
    --output data/processed
```

Run literal baseline:

```bash
python -m src.baselines.literal_reduction \
    --input examples/source.musicxml \
    --output outputs/musicxml/literal.musicxml
```

Train:

```bash
python -m scripts.train_model \
    --config configs/model/seq2seq.yaml
```

Generate:

```bash
python -m src.inference.generate \
    --input examples/source.musicxml \
    --preset romantic \
    --num_candidates 8 \
    --output_dir outputs/demo
```

Evaluate:

```bash
python -m scripts.evaluate_all \
    --pred outputs/demo \
    --reference examples/target.musicxml
```

---

## 22. Initial Configuration

Example `configs/model/seq2seq.yaml`:

```yaml
model:
  architecture: encoder_decoder_transformer
  d_model: 384
  n_heads: 6
  encoder_layers: 6
  decoder_layers: 6
  dropout: 0.1

representation:
  max_bars: 16
  subdivision: 24
  include_velocity: true
  include_articulation: false
  include_dynamics: false

training:
  batch_size: 8
  gradient_accumulation: 4
  learning_rate: 0.0002
  warmup_steps: 2000
  label_smoothing: 0.1

constraints:
  melody_mode: soft
  octave_equivalent: true

generation:
  beam_size: 4
  num_candidates: 8
```

These values are placeholders and should be tuned after measuring actual sequence lengths and hardware limits.

---

## 23. Development Milestones

### Milestone 0 — Repository Skeleton

Deliverables:

- repository structure
- environment
- test framework
- `PROJECT.md`
- minimal README

Success condition:

```text
pytest passes
```

### Milestone 1 — MusicXML Round Trip

Implement:

```text
MusicXML
-> normalized internal score
-> MusicXML
```

Success criteria:

- pitch preserved
- onset preserved
- duration preserved
- measure structure preserved

### Milestone 2 — Role-Aware Source Representation

Implement:

```text
solo extraction
accompaniment extraction
tokenization
detokenization
```

Success criterion:

A source example can be converted to tokens and reconstructed.

### Milestone 3 — Pair Alignment

Implement:

```text
measure alignment
beat alignment
melody note alignment
```

Visualize aligned notes.

Do not proceed to model training until alignment is inspectable.

### Milestone 4 — Transcription Analysis

Implement technique metrics.

Produce a table such as:

```text
pair_id
melody_preservation
octave_shift_rate
octave_doubling_rate
register_expansion
density_change
bass_expansion
added_note_ratio
```

This is the first meaningful research result.

### Milestone 5 — Literal and Rule-Based Baselines

Produce playable MIDI/MusicXML.

This establishes a lower bound.

### Milestone 6 — Vanilla Seq2Seq Model

Train on short segments first:

```text
4-8 measures
```

Goal:

Generate syntactically valid piano token sequences.

### Milestone 7 — Melody Constraints

Implement:

```text
none
soft
hard
```

Compare metrics.

### Milestone 8 — Technique Conditioning

Add control tokens and controlled generation.

Goal:

The same source should produce measurably different outputs when controls change.

### Milestone 9 — Reranking

Generate multiple candidates and rerank with:

- melody preservation
- playability
- technique match
- optional MusicBERT similarity

### Milestone 10 — Demo

Input:

```text
MusicXML
```

Controls:

```text
preset
virtuosity
melody preservation
octave doubling
register expansion
chord thickening
figuration
```

Output:

```text
score
MIDI
audio
analysis metrics
```

---

## 24. MVP Definition of Done

The first publishable/demo-ready version is complete when:

- [ ] MusicXML input works.
- [ ] Solo and accompaniment are explicitly represented.
- [ ] At least one paired training corpus is available.
- [ ] Alignment can be visualized and manually checked.
- [ ] Literal reduction baseline works.
- [ ] Rule-based pianistic baseline works.
- [ ] Seq2seq model produces valid symbolic piano.
- [ ] Generated melody is recognizably preserved.
- [ ] Output can be exported as MusicXML and MIDI.
- [ ] Playability heuristics are reported.
- [ ] At least three objective metric groups are implemented.
- [ ] At least one controlled technique is demonstrably effective.
- [ ] Human examples are rendered for listening.
- [ ] Results are compared against baselines.

A good initial controlled technique is:

```text
octave_doubling
```

because it is easy to detect, easy to interpret, and musically meaningful.

---

## 25. Recommended Implementation Order

Do **not** start by implementing a large model.

Recommended order:

```text
1. collect 5-10 clean pairs
2. MusicXML parser
3. role extraction
4. source-target alignment
5. visual inspection tools
6. transcription technique statistics
7. literal baseline
8. rule-based baseline
9. tokenizer
10. tiny Transformer overfit test
11. real train/validation split
12. melody constraint
13. technique conditioning
14. reranking
15. larger dataset
16. MusicBERT integration
17. demo
```

The critical sanity test is:

> Can a tiny model overfit 10-50 aligned segments?

If not, the representation or alignment is probably wrong.

---

## 26. Testing Strategy

### Unit Tests

Test:

- MusicXML parsing
- tied notes
- tuplets
- pitch conversion
- measure mapping
- source-role extraction
- token round trip
- melody alignment
- octave-equivalent matching
- density metrics
- register metrics

### Golden Score Tests

Keep a few tiny hand-verified scores.

For each:

```text
input MusicXML
expected normalized JSON
expected tokens
expected metrics
```

### Model Smoke Test

A tiny model must overfit a tiny dataset.

Expected:

```text
training loss -> very low
generation reproduces target structure
```

### Export Test

Every generated token sequence must either:

1. produce valid MusicXML/MIDI, or
2. fail with an explicit validation error.

---

## 27. Risks

### Risk 1 — Too Few Paired Scores

Mitigation:

- phrase-level segmentation
- transposition augmentation
- pretraining on generic symbolic music
- weakly paired mining
- self-supervised source reconstruction
- rule-generated pseudo-targets for warm start

### Risk 2 — Poor Alignment

Mitigation:

- hierarchical alignment
- visual debugging
- manual correction of validation/test pairs
- confidence score for every aligned segment

### Risk 3 — Model Learns Literal Copying

Mitigation:

- compare transformation statistics
- explicitly condition on technique
- train on artistically rich targets
- include a transformation-density objective or sampling policy

### Risk 4 — Model Destroys Melody

Mitigation:

- role-aware input
- melody auxiliary loss
- constrained decoding
- candidate reranking

### Risk 5 — Output Is Unplayable

Mitigation:

- rule-based playability scorer
- N-best reranking
- span/density constraints
- eventual learned playability model

### Risk 6 — Dataset Copyright / Licensing

Mitigation:

- store license metadata per pair
- prioritize public-domain sources
- never assume an online score is safe to redistribute
- release only data that has been individually checked
- provide scripts/metadata rather than copyrighted score files where necessary

---

## 28. Stretch Goals

### V2 — Jazz Arrangement

Add:

- extended chords
- guide-tone movement
- chord substitutions
- stride
- walking bass
- block chords
- drop-2-like pianistic voicing concepts
- rhythmic displacement
- fills

This may require a separate dataset and stronger harmony representation.

### V2 — Discrete Denoising / Masked Generation

Replace or augment autoregressive decoding:

```text
melody scaffold
+ masked piano texture
-> iterative filling
```

Research question:

> Does structure-aware masked generation better preserve global texture and melody than autoregressive generation?

### V2 — Arranger Representation

Learn an arranger embedding from transformation statistics.

Instead of saying:

```text
STYLE = LISZT
```

learn:

```text
arranger embedding
-> observable technique profile
```

### V3 — Orchestra / Ensemble to Piano

Input:

```text
solo + orchestra
```

Output:

```text
concert piano transcription
```

Requires source reduction and salience modeling.

### V3 — Audio Front End

```text
audio
-> transcription / source representation
-> VirtuosoTranscribe
-> piano solo
```

Keep this separate from the core arrangement model.

### V3 — Paraphrase / Fantasy Mode

Permit:

- theme reordering
- inserted transitions
- cadenzas
- harmonic reinterpretation
- large-scale formal freedom

This becomes a different research problem:

> theme-conditioned composition rather than faithful transcription.

---

## 29. Potential Paper / Portfolio Titles

Conservative:

> **Expressive Piano-Solo Transcription from Solo-and-Accompaniment Scores**

Research-oriented:

> **Melody-Preserving Neural Piano Transcription with Technique-Conditioned Symbolic Generation**

More distinctive:

> **VirtuosoTranscribe: Controllable Pianistic Rewriting of Solo-and-Accompaniment Scores**

If masked generation is later added:

> **Structure-Aware Denoising Transformers for Expressive Piano Transcription**

---

## 30. Recommended First Week

### Day 1

- initialize repository
- install `music21`
- collect 3-5 source/target pairs
- define metadata schema

### Day 2

- parse source and target MusicXML
- identify parts
- dump normalized notes to JSON

### Day 3

- implement solo melody extraction
- visualize source solo vs target piano piano-roll

### Day 4

- implement octave-tolerant melody alignment
- inspect results manually

### Day 5

- implement:
  - melody preservation
  - register expansion
  - note density
  - polyphony
  - octave doubling

### Day 6

- implement literal reduction baseline
- export MusicXML and MIDI

### Day 7

Create the first analysis report:

```text
Source -> Human Transcription
```

with:

- aligned score excerpt
- transformation statistics
- observed arrangement techniques

Only after this should neural model work begin.

---

## 31. Immediate Coding Tasks

Create these files first:

```text
src/data/parse_musicxml.py
src/data/normalize_score.py
src/analysis/melody_alignment.py
src/analysis/technique_features.py
src/baselines/literal_reduction.py
tests/test_parser.py
tests/test_alignment.py
```

First API target:

```python
score = parse_musicxml(path)

source = normalize_source(
    score,
    solo_part="Violoncello",
    accompaniment_parts=["Piano"]
)

features = extract_source_features(source)
```

Alignment target:

```python
alignment = align_solo_to_piano(
    source_solo=source.solo,
    target_piano=target
)
```

Analysis target:

```python
stats = analyze_transcription(
    source=source,
    target=target,
    alignment=alignment
)
```

Expected result:

```python
{
    "melody_coverage_exact": 0.86,
    "melody_coverage_octave_tolerant": 0.97,
    "octave_shift_rate": 0.18,
    "octave_doubling_rate": 0.24,
    "register_expansion_semitones": 14,
    "note_density_ratio": 1.41,
    "polyphony_delta": 1.7
}
```

These numbers are illustrative only.

---

## 32. Technical Stack

Recommended starting environment:

```text
Python 3.11+
PyTorch
music21
pretty_midi
mido
numpy
scipy
pandas
matplotlib
pytest
PyYAML / OmegaConf
```

Optional:

```text
transformers
miditok
muspy
pdmx
wandb
tensorboard
```

External score rendering:

```text
MuseScore
```

Optional audio rendering:

```text
FluidSynth + piano SoundFont
```

---

## 33. Useful Existing Work

### MusicBERT

Symbolic music representation model using OctupleMIDI and bar-level masking. It is best treated here as a representation model, classifier, or reranker rather than automatically assuming it should be the main generator.

- https://github.com/microsoft/muzic/tree/main/musicbert

### PDMX

Large public-domain-oriented MusicXML dataset for symbolic music processing. It contains more than 250K MusicXML scores and is a promising source for instrumentation/arrangement mining. Use the current license-clean subsets and inspect metadata carefully.

- https://github.com/pnlong/PDMX
- https://zenodo.org/records/15571083

### Pop2Piano

Useful prior work demonstrating that automatic piano-cover generation can be modeled as a generation problem. Its task is audio-to-piano, whereas this project focuses on explainable score-level transformation.

- https://github.com/sweetcocoa/pop2piano

### music21

Python toolkit for symbolic score analysis with MusicXML and MIDI import/export.

- https://music21.org/music21docs/

---

## 34. Core Design Principle

The project should not be evaluated only by whether it "sounds nice."

The central question is whether the model learns an interpretable transformation:

```text
SOURCE MUSICAL IDENTITY
        |
        | preserve
        v
MELODY / HARMONY / FORM
        |
        | transform
        v
REGISTER / VOICING / TEXTURE / FIGURATION
        |
        v
IDIOMATIC PIANO SOLO
```

A successful model should be able to answer both:

> **What did the arrangement preserve?**

and

> **What did the arrangement change?**

That interpretability is one of the strongest ways to distinguish this project from a generic MIDI generator.

---

## 35. Final MVP Recommendation

Build the first serious model around:

> **solo + piano accompaniment -> piano solo**

using:

1. role-aware symbolic input,
2. encoder-decoder Transformer,
3. melody-preservation constraints,
4. a small interpretable technique-control vector,
5. rule-based playability scoring,
6. N-best candidate reranking.

Do not begin with imitation of a named arranger.

First learn the underlying transformation operators.

Once the model can reliably preserve melody while rewriting texture, add:

```text
historical arranger analysis
-> technique distributions
-> arranger-conditioned presets
```

This creates a clean progression from an implementable engineering project to a serious MIR research project.
