# Recognition Module

這個模組放目前主力的手語辨識系統，重點是 fixed sentence 路線。

## 目前主軸

- daily30 fixed-sentence BiGRU
- MediaPipe pose + hand landmarks 特徵
- offline evaluation
- realtime sentence inference
- auto-trigger 邊界切段實驗

## 子目錄

- `training/`: 訓練腳本與模型訓練流程
- `evaluation/`: offline baseline 與 realtime-like proxy 評估
- `realtime/`: webcam / mp4 即時推論與 UI
- `inference/`: 共用特徵、模型與 artifact 載入
- `legacy/`: 舊版 baseline、早期流程與歷史參考

## 目前代表性內容

- `realtime/realtime_infer_daily30_sentence.py`
- `training/train_daily30_sentence_bigru.py`
- `evaluation/eval_daily30_sentence_bigru.py`
- `inference/build_daily30_sentence_manifest.py`
- `inference/daily30_sentence_feature_utils.py`
- `inference/daily30_sentence_model_utils.py`

## 備註

這個 preview repo 先保留代表性入口與共用程式，不把大量一次性 debug / remote 監控腳本全部放進來。
