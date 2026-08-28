# Runtime Models

MediaPipe 的 `.task` 模型檔放這裡。**本 repository 不散布這兩個檔案**
(見 README §4.3 公開範圍),請自行取得後放入:

- `hand_landmarker.task`
- `pose_landmarker.task`

## 必須用 lite,不是 full

訓練特徵是用 **`pose_landmarker_lite`** 抽取的,不是 `pose_landmarker_full`。
每一個特徵快取 `.npz` 都記著當時的模型雜湊,可以據此核對:

| 檔案 | SHA-256 |
|---|---|
| `hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `pose_landmarker.task`(lite) | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

用錯版本不會報錯,只會讓準確率無聲下降——landmark 分布跟訓練時不一致。

## 自動取得

```bash
python scripts/fetch_mediapipe_models.py
```

腳本會從 Google 官方位置下載,並用上表的 SHA-256 核對。**對不上就刪檔並報錯**,
不會留下一個看起來能用、實際上分布不對的模型。

手動核對:

```bash
sha256sum models/hand_landmarker.task models/pose_landmarker.task
```

正式版本直接從這個目錄載入;Windows 可攜版則從內附的 `resources/models/` 載入。
路徑可用環境變數 `SLR_MODELS_DIR` 覆寫。
