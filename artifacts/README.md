# Runtime Artifacts

這裡放即時辨識載入的模型 bundle。目前只有一個:

```
artifacts/
└── realtime/best_current/     42 類 Transformer
```

## `realtime/best_current/`

42 類 Knee42 Transformer，由
[`recognition.transformer.recognizer`](../recognition/transformer/recognizer.py) 載入。

| 檔案 | 內容 |
|---|---|
| `best_model.pt` | 權重（4 層 Transformer encoder，輸入 `[64, 657]`） |
| `feature_config.json` | 特徵合約：位置＋速度＋加速度三通道，不串遮罩、不做標準化 |
| `runtime_config.json` | 執行期合約：序列長度、219 維幀合約、輸入串流型別 |
| `label_map_knee42.json` | `K42_01`–`K42_42` ↔ index |
| `display_text_map.json` | label_id → 中文顯示字串 |
| `model_card.json` | 訓練切分、可宣稱的指標、已知限制 |
| `integrity_manifest.sha256` | 以上每個檔案的 SHA-256 |

載入時會逐項驗證雜湊與合約，任何一項對不上就拋 `IntegrityError` 拒絕啟動。
JSON 以「CRLF 正規化為 LF」之後計算雜湊，因此 Windows 與 Linux checkout
驗證結果一致。

**重建方式**（可重現，不需要原始影片）：

```bash
python scripts/build_knee42_transformer_bundle.py \
    --checkpoint <knee42_final_v2.pt> \
    --label-map <label_map_knee42.json> \
    --display-map <display_text_map.json> \
    --metrics docs/evaluation/knee42_loso_metrics.json \
    --out artifacts/realtime/best_current
```

### 關於這顆權重的成績，請先讀 `model_card.json`

發布的權重是在方法通過留一簽者驗證之後，**用全部四位簽者重新訓練**的版本，
因此它**沒有任何保留測試分數**。checkpoint 內的 `val_macro_mixed = 1.0`
來自不分簽者的隨機切分，屬於樂觀值，**不可當作準確率引用**。

可以引用的是方法本身的留一簽者結果，數字全部由
[`scripts/aggregate_knee42_loso_runs.py`](../scripts/aggregate_knee42_loso_runs.py)
從原始訓練 log 重算，存放於
[`docs/evaluation/knee42_loso_metrics.json`](../docs/evaluation/knee42_loso_metrics.json)。

## 路徑覆寫

| 環境變數 | 覆寫對象 |
|---|---|
| `SLR_RUNTIME_BUNDLE_DIR` | bundle 位置 |
| `SLR_MODELS_DIR` | MediaPipe `.task` 位置 |

未設定時的預設值見 [`recognition/config.py`](../recognition/config.py)。
