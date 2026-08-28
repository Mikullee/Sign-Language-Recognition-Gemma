# Recognition Module

42 類固定句型台灣手語辨識。**現行路徑是 Transformer**,BiGRU 保留為 legacy。

## 子目錄

| 目錄 | 內容 |
|---|---|
| `transformer/` | **現行辨識路徑**:特徵組裝(219 → 657)、模型、bundle 驗證推論、離線影片與切段 |
| `realtime/` | 相機擷取、auto-trigger 狀態機、**共用的幀層前處理**、legacy BiGRU 即時推論 |
| `training/` | Transformer 與 legacy BiGRU 的訓練 |
| `evaluation/` | 選模、一次性測試、auto-trigger 邊界校準 |
| `inference/` | legacy BiGRU 的分類器定義 |

## 兩條路徑的分界

幀層合約共用,實作於
[`realtime/knee42_preprocessing.py`](realtime/knee42_preprocessing.py):
MediaPipe 結果 → 移除 pose 25/26 → 肩寬正規化 → **219 個值 + 219 個遮罩**。

分岔在序列組裝:

| | 現行 Transformer | Legacy BiGRU |
|---|---|---|
| 序列 | 內插缺值 → 重取樣 64 幀 → 串速度與加速度 | 標準化 → 補 0 → 串遮罩 |
| 輸入 | `[64, 657]` | `[64, 438]` |
| 進入點 | `transformer/recognizer.py` | `realtime/knee42_ivcam.py` |

**兩者的 checkpoint 不可互換**,bundle 各自帶 `feature_config.json` 並在載入時強制驗證。

## 代表性入口

- `transformer/recognizer.py` — bundle 驗證 + 推論
- `transformer/segmentation.py` — 離線影片切段與整段辨識
- `training/knee42_transformer.py` — 留一簽者 / 全簽者訓練
- `realtime/knee42_ivcam.py` — legacy BiGRU 即時辨識(需自備 v11 bundle)
- `evaluation/eval_auto_trigger_boundaries.py` — auto-trigger 邊界校準

## 備註

v0.1.0 時期的 27 類 daily30 子系統已於 v12 移除,理由見
[`docs/changelog.md`](../docs/changelog.md)。
