# Sequence 補強錄製計畫（Project Order, Gap 70）

本計畫目標：在不改變氣象主播參考順序的前提下，補齊低 token gloss 缺口。

## 錄製模板

1. `P01`：`今天 大陸 越來越 其他 早晚 保暖 做好 要`
- 次數：20

2. `P02`：`大陸 晚 做好`
- 次數：25

## 總量

- 需新增錄製：45 部 sequence

## 執行重點

1. 影片命名與路徑請依既有規則放入 `raw_sequences`。
2. 審片通過後再移入 `approved_sequences`。
3. 每支影片都要完成時間標註（start_sec/end_sec）。
4. 這批完成後，重訓 BiGRU 並跑固定測試。
