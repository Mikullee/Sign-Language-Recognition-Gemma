# Recognition Overview

## 流程

1. 輸入影片或即時攝影機畫面
2. 透過 MediaPipe 抽取 pose 與 hand landmarks
3. 做座標正規化、時間長度調整與特徵建構
4. 將固定長度序列送入 BiGRU 分類器
5. 輸出句子類別、top3 候選與即時畫面資訊

## 目前重點

- fixed sentence 比 gloss 更穩定
- offline baseline 與 realtime-like evaluation 需分開看
- realtime demo 不只看 top1，也要看 top3 與 trigger 邊界

## 核心檔案

- `recognition/training/train_daily30_sentence_bigru.py`
- `recognition/evaluation/eval_daily30_sentence_bigru.py`
- `recognition/realtime/realtime_infer_daily30_sentence.py`
- `recognition/inference/daily30_sentence_feature_utils.py`
- `recognition/inference/daily30_sentence_model_utils.py`
- `recognition/inference/build_daily30_sentence_manifest.py`
