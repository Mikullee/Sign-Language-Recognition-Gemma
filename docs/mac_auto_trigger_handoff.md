# Mac 自動切段交接包

這個 GitHub prerelease ZIP 是目前 Web 自動切段的「調校候選版」，目的在讓 Apple Silicon Mac 直接接手現場測試。分類模型已包含在套件內，**不需要重訓**。

## 安裝與啟動

解壓 ZIP、開啟 Terminal，進入解壓後的資料夾：

```bash
chmod +x scripts/setup_mac_auto_trigger.sh scripts/run_mac_auto_trigger.sh
./scripts/setup_mac_auto_trigger.sh
./scripts/run_mac_auto_trigger.sh
```

然後開啟 `http://127.0.0.1:8642`。服務只綁在本機 loopback，不會開放給區網或公網。

需求：Apple Silicon Mac、macOS 14 以上、Python 3.10–3.14。Intel Mac 不適用這份固定版本套件。

## 套件包含內容

- 目前 `fix/web-auto-trigger-live-tuning` 分支程式碼。
- `artifacts/realtime/best_current/` 的 Knee42 Transformer v12、42 類模型。
- Python 與瀏覽器使用的 MediaPipe task／WASM 資產。
- 三支原始測試影片：`你好.mp4`、`我肚子餓.mp4`、`晚安.mp4`。
- 人工起訖標註：`data/annotations/auto_trigger_three_videos.csv`。
- `MAC_PACKAGE_MANIFEST.json`，記錄來源 commit 與每個檔案的 SHA-256。

## 目前狀態與限制

這不是已完成現場驗收的正式版。現行候選設定仍曾在三支影片上出現 `timeout_finalize`、結束過晚或漏啟動，必須在 Mac 繼續調校。timeout 段應視為切段失敗，不能用它判斷分類模型好壞。

三支影片可以檢查一般的「休息 → 手語 → 回到休息」起訖，但畫面沒有清楚拍到膝蓋，不能證明坐姿、手回到膝蓋時會正確切斷。最終至少再錄 3–5 支膝蓋清楚入鏡的影片，為每支填寫人工 `start_sec` 與 `end_sec`。

這一階段只調整切段，不使用 J/Test 調參，也不需要重新訓練分類模型。切段正確但詞句辨識錯誤，應另列為模型問題。

## 建議測試順序

1. 先切到「手動」模式，確認攝影機、MediaPipe 與 42 類模型能出結果。
2. 切到「自動偵測」，固定休息姿勢完成校準。
3. 依序比 `你好、我肚子餓、晚安`，記錄開始、結束、原因及是否 timeout。
4. 用三支標註影片的時間作為回歸基準，不要用辨識對錯代替邊界判定。
5. 補錄膝蓋可見影片，確認回到膝蓋後約 0.3–0.8 秒完成切段且不立即誤啟動。

停止服務可在 Terminal 按 `Control-C`。
