# VirtuosoTranscribe

依照 [VirtuosoTranscribe_PROJECT.md](VirtuosoTranscribe_PROJECT.md) 實作的 solo + piano accompaniment → piano solo 研究工具。V1 的 M0–M10 工程流程已串接：解析、角色分離、階層對齊、分析、基線、single／dual Transformer、旋律約束、13 維 conditioning、reranking、音訊、實驗與互動 demo。

目前 checkpoint 是小資料 overfit 模型；能自由生成全部訓練片段，但 held-out 旋律音高準確率仍低，不能解讀為藝術編曲品質或一般化能力。完整狀態見 [implementation status](docs/implementation_status.md)、[dataset card](docs/dataset_card.md)、[neural inference](docs/NEURAL_INFERENCE.md)。

## 安裝

Python 3.11+；本工作區使用 Python 3.12 與 PyTorch CPU。

```powershell
uv sync --extra dev --extra model --python 3.12
uv run --extra dev --extra model pytest -q
uv run --extra dev --extra model ruff check .
```

沒有 uv 時：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m ensurepip
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e ".[dev,model]"
```

`uv.lock` 固定依賴；Windows／Linux 的 model extra 預設使用 CPU wheel。CUDA 環境可安裝相容的 PyTorch wheel，指定 `--device cuda`。

## Demo

```powershell
.\.venv\Scripts\python -m scripts.run_demo --port 8765 --checkpoint-root outputs/checkpoints
```

開啟 <http://127.0.0.1:8765>，上傳 MusicXML／MXL／MIDI、選擇 preset 或 checkpoint、調整技法與旋律模式，播放結果、檢查對齊、下載 MusicXML／MIDI／WAV。服務只監聽 loopback，資料留在本機。WAV 是合成鋼琴類音色，沒有採樣鋼琴或真人演奏。

## 資料與訓練

JSON／CSV manifest 的 source／target 授權必須個別標示 `CHECKED` 並保留依據。版本、移調與切段依 `work_id` 固定於同一 split。`examples/pairs.json` 是同作品的三個合成變體，全部在 train。

```powershell
.\.venv\Scripts\python -m scripts.prepare_dataset --manifest examples/pairs.json --output data/processed/demo

# 12 個原創小作品：10 train／1 validation／1 test
# conditioned corpus 有同一來源的 plain／octave-doubled 控制版本
.\.venv\Scripts\python -m scripts.make_training_data
.\.venv\Scripts\python -m scripts.train_model --config configs/train/tiny_overfit.yaml
.\.venv\Scripts\python -m scripts.train_model --config configs/train/tiny_conditioned.yaml

# 已查核的真實 corpus
.\.venv\Scripts\python -m scripts.train_model --config configs/model/conditioned.yaml --dataset data/processed/curated/dataset.json --output outputs/checkpoints/curated
```

訓練含 causal／padding mask、teacher forcing、AdamW、warmup、gradient clipping／accumulation、validation、seed、weights-only checkpoint 與 optimizer／RNG resume。soft melody loss 是可微分的 aligned-onset presence loss。`scripts/train_baseline.py` 是相容入口。

已查核的 Schubert／August Horn 真實配對與重現流程見 [dataset card](docs/dataset_card.md)。`configs/train/real_overfit.yaml`、`real_dual.yaml`、`real_conditioned.yaml` 的三個模型皆已實際訓練並自由生成重現 10／10 小節。這是同一作品內的 overfit 檢查，沒有真實 held-out works；完整結果在 `outputs/checkpoints/real_training_summary.json`。

實際 sanity checkpoints 自由生成完整重現 vanilla 10／10、dual 10／10、conditioned 20／20 個片段，沒有 target prefix、查表或修補。完整結果與重現命令位於 `outputs/checkpoints/training_summary.json`。Validation pitch-token accuracy 約 44–50%；syntax-token accuracy 不能代表旋律正確。

## 生成與分析

```powershell
.\.venv\Scripts\python -m src.baselines.literal_reduction --input examples/source.musicxml --output outputs/musicxml/literal.musicxml
.\.venv\Scripts\python -m src.inference.generate --input examples/source.musicxml --preset romantic --num_candidates 4 --output_dir outputs/baseline

.\.venv\Scripts\python -m src.inference.generate --input data/interim/training_fixture/work_001_source.musicxml --checkpoint outputs/checkpoints/tiny_conditioned/last.pt --melody-mode hard --temperature 0 --num_candidates 1 --output_dir outputs/neural

.\.venv\Scripts\python -m scripts.evaluate_all --pred outputs/baseline --source examples/source.musicxml --reference examples/target.musicxml --output outputs/evaluation/baseline.json
.\.venv\Scripts\python -m scripts.analyze_pair --source examples/source.musicxml --target examples/target.musicxml --output-dir outputs/analysis/demo
```

角色模糊時以 `--solo-part`、可重複的 `--accompaniment-part` 指定名稱或 ID。輸出包含雙譜表鋼琴譜、MIDI、canonical JSON、候選指標／排名、raw neural tokens 與對齊 HTML／JSON；固定 seed 可重現取樣，沒有 checkpoint fallback。

`none` 使用事件 grammar；`soft` 加入旋律 logit 偏好；`hard` 在解碼時要求原始起音、時值及原音高或 ±12 半音，記錄強制步驟。預算、context、grace-note 或 vocabulary 不相容時明確失敗。

Presets：`faithful`、`romantic`、`virtuosic`、`concert_transcription`。Rule-based 支援八度加倍、低音強化、音域、和弦加厚、內聲保留、分解和弦與密度；未實作的 melody rewriting／countermelody／ornamentation／reharmonization／rhythmic rewriting 會拒絕。Neural 接收全部 13 維連續控制，已實測學到的是 octave doubling，其餘維度需要相應訓練資料。

## 實驗與盲測

```powershell
.\.venv\Scripts\python -m scripts.run_experiments --manifest examples/pairs.json --split all --config configs/evaluation/full.yaml --checkpoint vanilla=outputs/checkpoints/tiny_vanilla/last.pt --checkpoint conditioned=outputs/checkpoints/tiny_conditioned/last.pt --checkpoint dual=outputs/checkpoints/tiny_dual/last.pt --output-dir outputs/experiments/full
.\.venv\Scripts\python -m scripts.human_evaluation prepare --report outputs/experiments/full/report.json --output-dir outputs/human_evaluation
```

涵蓋 B0／B1／B2／B3、single／dual、none／soft／hard、控制、原順位／rerank、reference。產生 CSV／JSON／HTML、樂譜與試聽音訊，記錄 checkpoint SHA256、in-sample overlap、資料種類、缺少的方法及失敗。Work-level bootstrap 需要至少兩個作品；小合成資料不能支持研究品質結論。

本工作區的真實譜比較在 `outputs/experiments/real/index.html`，45 筆皆成功；歷史人工參考的譜與 WAV 在 `outputs/real_examples/`，匿名試聽在 `outputs/human_evaluation_real/participant/index.html`。同一作品的節錄全部明確標記為訓練集診斷。

盲測只提供 `outputs/human_evaluation/participant` 給受試者；`private/key.json` 是方法對照。有真實填答後：

```powershell
.\.venv\Scripts\python -m scripts.human_evaluation aggregate --key outputs/human_evaluation/private/key.json --ratings actual_ratings.csv --output outputs/human_evaluation/ratings_summary.json
```

沒有評分時是 0 observations，不會產生虛構人類評分。

## 表示法與限制

Canonical onset／duration 單位為四分音符，pitch 為 concert MIDI，保留 pickup、ties、tuplets、key／meter／tempo／dynamics metadata。完整 token bundle 可重建表示法；裸 tokens 不含全部 notation。預設每四分音符 24 格，不相容節奏會拒絕，可調整 `--subdivisions`。長譜需先切成 checkpoint 支援的 context。

Playability 是持續音符、手跨度、跳躍與密度的符號代理，沒有 fingering、pedal 或真人驗收。MusicBERT 為 supplied fairseq／TorchScript encoder 的選用 adapter；未完成 genuine pretrained encoder 的本機實測。V2／V3 stretch goals 是後續研究，V1 不含 audio transcription、orchestral reduction 或自由形式重作。
