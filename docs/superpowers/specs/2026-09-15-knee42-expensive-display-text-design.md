# Knee42「很貴」顯示文字設計

## 目標

將 Knee42 類別 `K42_26` 的人類可讀文字由「太貴了」一致改為「很貴」，不改變辨識類別、class index、模型權重或 checkpoint 合約。

## 現況與邊界

- `label_map_knee42.json` 將 `K42_26` 固定映射到零起算 index `25`。
- Transformer checkpoint 與載入器使用 `K42_26` 類別 ID，而非中文顯示文字，驗證 42 類的順序。
- `display_text_map.json` 才是即時辨識、Web 介面及分析輸出的中文文字來源。
- 資料驗證器中的中文陣列只產生 `display_text` metadata；訓練仍以 `label_id` 建立 class index。

## 修改

將所有受版本控制的「太貴了」文字來源改為「很貴」：

1. runtime bundle 的 `display_text_map.json`；
2. `scripts/validate_knee42_data.py` 的 canonical display metadata；
3. `docs/evaluation/live_check_42.csv`；
4. README 的類別表與評估表。

因 runtime bundle 會驗證 SHA-256，更新 `integrity_manifest.sha256` 及 README 中 `display_text_map.json` 的 digest。`label_map_knee42.json`、`best_model.pt` 與其 digest 均保持原值。

## 相容性要求

- `K42_26` 必須仍映射至 index `25`。
- `idx_to_label[25]` 必須仍為 `K42_26`。
- checkpoint、模型權重與輸出維度不得修改。
- runtime bundle 必須通過現有完整性驗證。

## 測試

先加入會因舊文字而失敗的回歸測試，鎖定 `K42_26` 的新顯示文字與既有 index。修改後執行該測試、Transformer bundle 測試、release safety 測試及完整測試套件；完整套件若仍只有已確認的 Mac 計畫文件絕對路徑問題，將明確列為基線失敗。
