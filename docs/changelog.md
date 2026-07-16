# Changelog

## v0.1.0

- 正式支援 27 個固定句型，移除未受模型支援的 `T09 我聽不懂`
- 新增共用 auto-trigger 狀態機與秒制參數
- 加入 pre-roll、可見雙手腕、身體兩側、低動作與滑動投票句尾判斷
- 加入三支標註影片的離線校準、邊界評估、分類比較與 debug 影片
- Auto 模式改為站姿操作提示，可在冷卻後連續辨識下一句
- 移除遠端主機、帳號、密碼、SSH 自動下載與遠端執行參數
- runtime bundle、MediaPipe 模型與自動切段設定改為完全離線
- 新增 `app_config.json`，CLI 明確參數優先
- 新增 Windows x64 PyInstaller `onedir` 可攜版與 iVCam 啟動腳本
- 新增敏感資訊、27 類一致性、frozen 資源路徑與打包內容測試
