# Commit 規範

請使用結構化 commit 訊息，確保歷程可追蹤、可審查。

## 格式

`<type>: <short summary>`

範例：
- `data: 新增 09C-12C approved sequence 影片`
- `exp: 調整 mixed BiGRU 參數 lr=3e-4 wd=1e-4`
- `eval: 新增 Gate A 10 句測試結果表`
- `docs: 更新每週目標與流程規則`
- `fix: 修正 README 的 dev-agent-sync 連結`

## 類型（type）

- `data`：資料蒐集與標註變更
- `exp`：模型訓練與實驗變更
- `eval`：評估與報告變更
- `docs`：純文件變更
- `fix`：程式或邏輯修正
- `chore`：維護性、非功能性變更

## 範圍規則

- 一個 commit 只做一個原子變更。
- 避免使用 `Update ...` 這種無資訊的訊息。
