# Project Status

目前的辨識模型為 **42 類 Transformer（v12）**，應用層封裝仍為 `v0.1.0`。核心流程已完成：

- 42 類 Transformer runtime bundle，以 SHA-256 逐檔驗證後載入
- 42 類 v11 BiGRU 保留為 legacy（Release 提供，見 README「Legacy」）
- 27 類 daily30 子系統已於 v12 整套移除，Windows 可攜版隨之停止
- MediaPipe pose/hand 特徵抽取
- 共用的秒制 auto-trigger 狀態機
- 三支標註影片的自動起訖離線校準與 debug 輸出
- Auto 模式站姿提示與連續兩句重新偵測
- 完全離線載入，不含 SSH 主機、帳號、密碼或自動下載
- Windows x64 PyInstaller `onedir` 打包設定

現行 42 類模型包含 `K42_09 我聽不懂`。

已知限制：

- **發布權重沒有獨立保留測試分數。** 它在方法通過留一簽者驗證後，用全部四位簽者重新訓練，
  因此四位簽者都無法再用來量測；signer J 的一次性額度已由 legacy BiGRU 消耗。詳見 README §2.3。
- 留一簽者結果在 signer X 上明顯偏低（macro top-1 .729），跨簽者穩定性不足。
- 目前訓練資料主要為坐姿，站姿可能造成分類分布偏移。
- 自動起訖第一版只針對目前拍攝者、攝影機位置與距離校準。
- 乾淨 Windows 電腦與實際 iVCam 的最終人工驗收仍需在目標設備執行。
