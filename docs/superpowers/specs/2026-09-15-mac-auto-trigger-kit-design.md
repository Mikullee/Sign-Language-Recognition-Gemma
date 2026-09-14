# Knee42 Mac 自動切段交接包設計

日期：2026-09-15  
基準分支：`fix/web-auto-trigger-live-tuning`  
發布型態：私人資料 repository 的 GitHub prerelease 單一 ZIP

## 目標

讓 Apple Silicon Mac 從 GitHub 下載一個 ZIP 後，可安裝 Python 相依套件、驗證內附模型與 MediaPipe 資產、啟動本機 Web 攝影機頁面，並立即使用三支既有人工標註影片繼續自動切段調校。

## 範圍

- 保留目前 Transformer v12、42 類模型，不重訓分類模型。
- 打包目前 Web 自動切段候選分支，不宣稱切段已完成驗收。
- 私人 Release ZIP 包含 Git 追蹤的原始碼、模型 bundle、兩個 Python MediaPipe `.task`、瀏覽器 MediaPipe WASM、三支人工標註影片及 CSV。
- Git 儲存庫不直接追蹤第三方二進位資產與影片；它們只存在於 Release 資產。
- 完整 ZIP 只發布到 `Knee42-Private-Reproduction-Data`；公開程式 repository 不新增人物影片或 MediaPipe 二進位。
- 提供 Mac 安裝、驗證與啟動腳本，以及一份已知限制與後續工作說明。

## 使用流程

1. 從 GitHub prerelease 下載並解壓 ZIP。
2. 在 Terminal 執行 `./scripts/setup_mac_auto_trigger.sh`。
3. 執行 `./scripts/run_mac_auto_trigger.sh`。
4. 開啟 `http://127.0.0.1:8642`，先用手動模式確認模型，再用自動模式調整切段。
5. 以 `data/annotations/auto_trigger_three_videos.csv` 作為既有邊界基準；另錄膝蓋清楚入鏡的片段做最終驗收。

## 安全與失敗處理

- 安裝腳本只支援 macOS，並明確拒絕 Intel Mac，因固定版本的 MediaPipe 套件以 Apple Silicon 為目標。
- 啟動前驗證 Python 版本、模型 bundle 完整性、MediaPipe 資產與三支測試影片。
- Web 服務只綁定 `127.0.0.1` 並使用 localhost HTTP，不對區網或公網開放。
- Release 標記為 prerelease；README 明列目前三支影片不包含清楚膝蓋畫面，且現行候選仍有 timeout／漏啟動待調校。

## 驗收標準

- 自動測試驗證打包清單、必要檔案及 Mac 腳本的保護條件。
- 打包工具能從乾淨的 Git 提交建立 ZIP，並拒絕缺檔或雜湊錯誤的資產。
- ZIP 解壓後的內部驗證命令通過，模型 bundle 維持 42 類。
- 完整專案測試通過，分支推送 GitHub，prerelease 含 ZIP 與 SHA-256 檔。
