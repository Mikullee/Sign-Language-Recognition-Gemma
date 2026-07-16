# Project Status

目前正式版本為 `v0.1.0`，核心流程已完成：

- 27 類固定句型 BiGRU runtime bundle
- MediaPipe pose/hand 特徵抽取
- 共用的秒制 auto-trigger 狀態機
- 三支標註影片的自動起訖離線校準與 debug 輸出
- Auto 模式站姿提示與連續兩句重新偵測
- 完全離線載入，不含 SSH 主機、帳號、密碼或自動下載
- Windows x64 PyInstaller `onedir` 打包設定

`T09 我聽不懂` 不在目前模型標籤中，因此文件與模板均不列為可辨識句型。

已知限制：

- 目前訓練資料主要為坐姿，站姿可能造成分類分布偏移。
- 自動起訖第一版只針對目前拍攝者、攝影機位置與距離校準。
- 乾淨 Windows 電腦與實際 iVCam 的最終人工驗收仍需在目標設備執行。
