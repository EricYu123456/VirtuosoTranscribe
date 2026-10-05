# Learned generation and melody constraints

The inference command loads a Transformer state dictionary and its exact training
vocabulary. It encodes the source once, then predicts every target token using
causal decoder logits. Generation never retrieves a training target or invokes a
baseline as a fallback. Missing, incompatible, corrupt, or unsuccessful neural
checkpoints produce explicit errors.

```powershell
.venv/Scripts/python.exe -m src.inference.generate `
  --input data/interim/training_fixture/work_001_source.musicxml `
  --solo-part Violin `
  --checkpoint outputs/checkpoints/tiny_vanilla/last.pt `
  --melody-mode hard --temperature 0 --num-candidates 1 `
  --output-dir outputs/my_learned_arrangement
```

`--temperature 0` selects greedy decoding. Positive temperatures use seeded
nucleus sampling controlled by `--top-p`. `--beam-width` greater than one uses
beam search and returns distinct completed token paths for the requested N-best
list. Musical fingerprints can still coincide, for example when paths differ
only in velocity; the output explicitly counts distinct musical candidates.
Source token streams and decoder prefixes must fit the checkpoint context. Long
scores require measure or phrase segmentation; the command never truncates them
silently. The default two CPU threads avoid excessive small-matrix overhead.

## Constraint semantics

- `none`: the event grammar checks syntax, chronological positions, source bar
  metadata, piano range, and positive durations. It provides no melody protection.
- `soft`: adds finite positive logit biases at known source melody attacks. It
  can generate missing or changed melody notes; it never inserts them afterward.
  Training can independently enable the differentiable melody-presence loss.
- `hard`: protects every annotated solo attack, including simultaneous attacks
  and repetitions. Each protected attack must have a target note at the original
  position, with the original duration and pitch equal to the source pitch or
  exactly one octave above or below. Existing bass-before-melody chord order can
  remain intact. If the model tries to advance past an unsatisfied attack, the
  decoder forces that event before advancing.

Hard mode performs no final melody projection. It either satisfies all attacks
during decoding or raises an error. A finite token budget also prevents endless
event repetition. Grace notes need notation sidecars and are explicitly rejected
by hard generation; bare training tokens cannot reconstruct those sidecars.
Complete source measure, meter, key, tempo, and dynamic metadata accompanies the
output. Generated voice IDs survive, while register assigns the notated piano
staff; this does not claim a learned fingering or hand-assignment model.

Every candidate exports MusicXML, MIDI, normalized JSON, the model's musical
tokens before projection, the same final tokens, and a decoding report. The
report records the seed, real model log probability, constrained log probability,
protected-event count, and zero projection repairs. The chosen output additionally
includes melody alignment, visualization, and all symbolic metrics.

## Technique conditioning and reranking

All thirteen spec controls are accepted as continuous neural inputs in `[0,1]`.
Named presets and `--control NAME=VALUE` overrides use the same ordered vector as
training. A vanilla checkpoint ignores conditioning and reports that fact;
controls then affect reranking only. Dimensional wiring does not establish an
artistic effect without training examples varying that dimension. The checked-in
overfit fixture verifies a learned octave-doubling response, rather than a claim
that every technique has already been learned from artistic transcriptions.

```powershell
.venv/Scripts/python.exe -m src.inference.generate `
  --input data/interim/training_fixture/work_001_source.musicxml `
  --checkpoint outputs/checkpoints/tiny_conditioned/last.pt `
  --melody-mode soft --control octave_doubling=1 `
  --num-candidates 4 --seed 42 --output-dir outputs/my_conditioned_candidates
```

Reranking combines melody coverage, exact pitch coverage, duration-weighted
harmonic consistency, playability heuristics, and normalized technique matches.
Its proxy scales match training's technique extraction. Countermelody and
ornamentation lack reliable labels in the initial corpus and are excluded from
technique-match scoring rather than receiving invented measurements. Every
weight, component, observation, and limitation is saved with the ranking.

## Optional pretrained MusicBERT scoring

`src.inference.musicbert.MusicBERTContentScorer` supports a supplied genuine
MusicBERT encoder and dictionary. The upstream fairseq backend additionally
requires the model registration directory and a compatible fairseq installation.
A TorchScript export can instead implement `forward(input_ids[B,S]) -> [B,S,D]`
or `[B,D]`. The adapter uses upstream OctupleMIDI bins and computes normalized
cosine similarity from real model embeddings. It never substitutes a random
encoder or heuristic score when a pretrained model is unavailable.

```powershell
.venv/Scripts/python.exe -m src.inference.generate `
  --input example.musicxml --checkpoint trained_transcription.pt `
  --musicbert-checkpoint exported_musicbert_features.pt `
  --musicbert-dictionary path/to/dict.txt `
  --musicbert-backend torchscript --output-dir outputs/musicbert_ranked
```

The adapter contract and actual tensor execution are unit-tested with an
explicitly labeled test feature model. A pretrained MusicBERT checkpoint is not
bundled, and no pretrained semantic-score quality claim is made without one.
The encoding and installation reference is the
[upstream MusicBERT repository](https://github.com/microsoft/muzic/tree/main/musicbert).
