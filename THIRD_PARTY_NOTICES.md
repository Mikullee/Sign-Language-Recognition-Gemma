# 第三方元件授權聲明

本專案使用下列第三方軟體。各元件維持其原始授權，不受本 repository 的 MIT License 影響。實際版本以 `requirements.lock.txt` 為準。

| 元件 | 用途 | 授權 | 來源 |
|---|---|---|---|
| MediaPipe | Pose 與 Hand 關鍵點抽取 | Apache License 2.0 | https://github.com/google-ai-edge/mediapipe |
| PyTorch | BiGRU 模型訓練與推論 | BSD-3-Clause | https://github.com/pytorch/pytorch |
| OpenCV (`opencv-python`) | 影像擷取、旋轉 metadata 處理、錄影 | Apache License 2.0（OpenCV 4.5.0 起；更早版本為 BSD-3-Clause） | https://github.com/opencv/opencv |
| NumPy | 陣列運算與特徵處理 | BSD-3-Clause | https://github.com/numpy/numpy |
| Pillow | 介面中文字繪製 | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow |

## MediaPipe 模型檔（重要）

本專案的 Windows 套件與 repository **均不隨附** MediaPipe 的模型檔：

```text
hand_landmarker.task
pose_landmarker.task
```

上述 `.task` 檔為 Google 發布的預訓練模型資產，其散布條款以 Google 官方發布頁與各模型的 model card 為準，與 MediaPipe 原始碼的 Apache 2.0 授權**未必相同**。為避免轉散布問題，Google 官方資產另行提供，不屬於本 repository 或本次 Release；如有需要請聯絡專案維護者取得說明。

取得後由 `model/integrity_manifest.sha256` 驗證雜湊，確保檔案與開發時一致。

## 字型

介面中文顯示依序嘗試 Windows 系統字型 `msjh.ttc`（微軟正黑體）、`msyh.ttc`、`mingliu.ttc`。**本專案不散布任何字型檔**，僅在執行時載入使用者作業系統既有的字型。找不到可用字型時會顯示明確診斷訊息，而非輸出亂碼。

## 補充

上表所列授權為各專案於撰寫時的公開條款。實際採用版本鎖定於 `requirements.lock.txt`；若升級任一元件，請一併確認其授權是否變更，並更新本檔案。各授權全文請參閱上表的來源連結。
