# Sign Language Recognition Gemma

這是固定句型手語辨識的正式離線版本。系統以 MediaPipe 擷取姿態與雙手特徵，使用 27 類 BiGRU 模型辨識一句完整手語，並透過已校準的自動起訖切段，在使用者開始動作時啟動、雙手回到身體兩側且穩定約 0.5 秒後完成辨識。

## v0.1.0 功能

- 27 個固定句型；模型不支援 `T09 我聽不懂`
- 自動開始、句尾確認、冷卻後連續辨識下一句
- 約 0.20 秒 pre-roll，避免漏掉開頭
- 句尾同時檢查雙手腕可見、雙手位於身體兩側與低動作
- 單幀手腕漏偵測不會直接結束
- 三支標註影片校準結果皆在 ±0.30 秒內
- 模型、MediaPipe 資源與自動切段設定皆隨程式離線載入
- Windows x64 `onedir` 可攜版不需要 Python、SSH 或網路

## 操作姿勢

Auto 模式以站姿為預設：

1. 面向攝影機站立，確保上半身、雙手腕與身體兩側都在畫面內。
2. 開始前雙手自然垂放身側。
3. 完成一句後將雙手放回身側並維持約 0.5 秒。
4. 畫面回到「可開始下一句」後即可連續辨識。

坐姿資料可能使分類模型對站姿產生分布差異；目前版本的自動切段以站姿降低「手不知道放哪裡」造成的句尾不穩定。若站姿分類準確率不足，應補錄站姿資料或混合姿勢再訓練，而不是放寬句尾規則。

## 支援句型

`T01–T08`、`T10–T23`、`T25`、`T27–T30`，共 27 類。完整文字請見 [`fixed_sentence_templates_daily30.csv`](artifacts/legacy/daily30_27class/fixed_sentence_templates_daily30.csv)。

## 快速開始

原始碼方式：

```powershell
conda env create -f environment.yml
conda activate slr_runtime
python -m recognition.realtime.realtime_infer_daily30_sentence
```

或使用：

```powershell
.\scripts\run_realtime_daily30_sentence.ps1
```

Windows 可攜版解壓後，可直接執行 `start_ivcam.cmd` 或 `SignLanguageRecognition.exe`。預設攝影機索引、backend、trigger mode 與日誌設定在 [`app_config.json`](app_config.json)；CLI 明確指定的值優先。

## 離線資源

- `models/hand_landmarker.task`
- `models/pose_landmarker.task`
- `artifacts/legacy/daily30_27class/best_model.pt`
- `artifacts/legacy/daily30_27class/label_map_v1.json`
- `artifacts/legacy/daily30_27class/fixed_sentence_templates_daily30.csv`
- `artifacts/legacy/daily30_27class/best_auto_trigger.json`

缺少任何必要檔案時程式會停止並指出缺少項目，不會建立 SSH 或其他網路連線。

## 測試與打包

```powershell
python -m unittest discover -s tests
python -m pip install -r requirements-build.txt
.\scripts\build_windows_portable.ps1
```

輸出位於 `dist/SignLanguageRecognition/`，ZIP 位於 `release/SignLanguageRecognition-v0.1.0-windows-x64.zip`。執行結果寫入可攜版旁的 `logs/`。

更多操作請見 [`docs/quick_start.md`](docs/quick_start.md)，目前狀態請見 [`docs/project_status.md`](docs/project_status.md)。
