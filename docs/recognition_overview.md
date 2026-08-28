# Recognition Overview

## 流程

1. 輸入影片或即時攝影機畫面
2. 透過 MediaPipe 抽取 pose 與 hand landmarks
3. 移除 pose 25/26、做肩寬相對正規化 → 每幀 219 個值 + 219 個遮罩
4. 沿時間軸內插缺失值、重取樣為 64 幀、串接速度與加速度 → `64 × 657`
5. 送入 4 層 Transformer encoder 分類器
6. 輸出句子類別、top3 候選與即時畫面資訊

第 1–3 步是兩條路徑共用的幀層合約；legacy BiGRU 路徑在第 4 步改為標準化後串接遮罩（`64 × 438`）。

## 目前重點

- fixed sentence 比 gloss 更穩定
- offline baseline 與 realtime-like evaluation 需分開看
- realtime demo 不只看 top1，也要看 top3 與 trigger 邊界

## 核心檔案

現行 Transformer 路徑：

- `recognition/transformer/features.py` — 序列組裝（219 → 657）
- `recognition/transformer/model.py` — Transformer encoder
- `recognition/transformer/recognizer.py` — bundle 驗證與推論
- `recognition/realtime/knee42_preprocessing.py` — 共用的幀層合約

Legacy：

- `recognition/training/train_daily30_sentence_bigru.py`
- `recognition/evaluation/eval_daily30_sentence_bigru.py`
- `recognition/realtime/realtime_infer_daily30_sentence.py`
- `recognition/inference/daily30_sentence_feature_utils.py`
- `recognition/inference/daily30_sentence_model_utils.py`
- `recognition/inference/build_daily30_sentence_manifest.py`
