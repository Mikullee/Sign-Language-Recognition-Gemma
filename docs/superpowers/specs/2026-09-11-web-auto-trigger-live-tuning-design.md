# Knee42 Web 自動切段即時調校設計

日期：2026-09-11  
基準：`main@c69bb9b95293bf6f7f1caee6273e59186ddc8c12`  
工作分支：`fix/web-auto-trigger-live-tuning`

## 目標與範圍

以 GitHub `main` 的 Transformer v12 Web Service 為唯一正式基準，建立可快速反覆執行的現場測試迴圈。使用者實際比劃、立即回報症狀；系統保留狀態與切段證據；每輪只調整一項設定或一個明確邏輯，再測試、重啟並比較結果。

本工作只改善自動啟動、結束、重新待命與切段邊界，不重訓或更換辨識模型，不使用已消耗的 J Test 做任何調參，也不把本機 `Knee42-IVCAM-20260819_082111` 當正式程式基準。該版本只作為使用體驗參考。

## 已知基線

- Web 現場測試 11 段中有 10 段以 12 秒 `timeout_finalize` 結束。
- 超時後約 0.68–0.70 秒再次啟動，幾乎等於 0.67 秒 cooldown，表示狀態機未真正回到休息姿勢。
- 開場建立的 rest reference 在整個 session 中不會更新。
- 手部只偵測到一側時，rest signature 可能無法計算。
- 最新 Web UI 已能顯示 `rest_distance`、`rest_threshold` 與手部偵測狀態，可直接作為除錯證據。

## 選擇的方案

採用「受監督的即時單變因迭代」，而不是直接憑感覺一次改多個閾值，也不先做大型離線網格搜尋。

每一輪流程：

1. 啟動最新 Web Service，確認 health、模型與 MediaPipe 資產正常。
2. 使用者依固定清單比劃 5–10 句，並回報每句正確答案及症狀。
3. 保存每段的開始／結束時間、收尾原因、rest distance、偵測手數、Top-3 與使用者標註。
4. 先區分「切段錯誤」和「模型分類錯誤」。只有前者進入本工作的修正範圍。
5. 每輪只更動一個可歸因因素，執行相關自動測試，再重啟服務。
6. 與前一輪同一組句子比較；改善才保留 commit，退步就回復該輪變更。

大型標註影片網格搜尋保留為第二階段方法；只有現場單變因調校仍無法穩定時才啟用。

## 修正順序

1. 將 `max_segment_sec` 從 12 秒縮短為符合 Knee42 句長的安全上限，先降低失敗片段污染。
2. 依現場 `rest_distance` 分布評估是否將 reference threshold 從 0.18 放寬；不得在沒有觀測值時定案。
3. 加入受控的 rest reference 重新校準路徑，避免開場姿勢永久鎖定。
4. 為單手缺失建立可驗證的 wrist／pose fallback；資訊不足時應顯示原因，不得靜默判成動作中。
5. 仍有問題時，才調整 `end_hold_sec`、vote ratio、hidden rest 與 cooldown。

## 元件與資料流

- `webservice/static/index.html`：擷取瀏覽器 MediaPipe landmark、顯示診斷值與使用者可見狀態。
- `webservice/server.py`：接收時間戳與 landmarks，回傳狀態、切段結果及 Top-3。
- `recognition/realtime/auto_trigger.py`：唯一的自動切段狀態機與 rest reference 管理者。
- `configs/auto_trigger_knee_v1.json`：基準設定；每個候選版本另存，禁止校準程序默默覆寫正式設定。
- 測試與 session log：保存每次迭代的設定摘要、症狀與量測，不保存或公開人物原始影像，除非使用者另行明確同意。

資料流為：瀏覽器影像 → 本機 MediaPipe landmarks → `/stream` → trigger state machine → 完整片段 → Transformer v12 → Top-3 → Web UI／session evidence。

## 錯誤處理

- 相機開啟但沒有 frame、模型或 MediaPipe 資產缺失時，停止測試並顯示可行修復，不繼續產生假結果。
- rest distance 無法計算時，記錄是左手、右手、雙手或 pose fallback 缺失。
- 到達 max segment 時保留 `timeout_finalize`，但標示為切段失敗，不拿該段分類結果判斷模型品質。
- 每次修改前保留可回復 commit；正式 `main` 不直接接受未通過實測與測試的設定。

## 驗證與完成標準

- 自動測試：先跑 trigger、Transformer realtime 與 Web Service 的聚焦測試，再跑完整 `pytest`。
- 現場驗收：固定句單至少 20 句，記錄正確答案與順序。
- 至少 90% 片段正常開始並以 rest 結束；目標為 95%。
- 不再出現 cooldown 後立即連續誤啟動。
- 一般動作結束後約 0.3–0.8 秒完成切段。
- 開頭與結尾未被肉眼可見地截斷，timeout 段不得計為成功。
- 切段正確但分類錯誤者另列模型問題，不藉 trigger 調參掩蓋。

## 發布方式

完成後以獨立 PR 合併到 `main`。更新 field notes、設定 provenance、測試結果與啟動說明後，再建立新的 GitHub Release；不覆寫舊的 `v1.0.0-v13` 資產。
