# Knee42 Web 最終交接版｜2026-09-15

版本標籤：`web-final-20260915`。使用者已在自己的 Chrome／攝影機環境試用並同意將目前版本作為最終版基準。此驗收限目前使用情境，不代表所有簽者或全部 42 句都已完成量化驗收。

這次交接只封存已驗收的程式，不再調整模型、辨識特徵或切段參數。

## 下載與選對版本

- [Release／下載頁](https://github.com/Mikullee/Sign-Language-Recognition-Gemma/releases/tag/web-final-20260915)
- [Web 完整原始碼＋模型 ZIP](https://github.com/Mikullee/Sign-Language-Recognition-Gemma/releases/download/web-final-20260915/Knee42-Web-Final-20260915.zip)
- [ZIP 的 SHA-256](https://github.com/Mikullee/Sign-Language-Recognition-Gemma/releases/download/web-final-20260915/Knee42-Web-Final-20260915.zip.sha256)
- [可審查的 PR #12](https://github.com/Mikullee/Sign-Language-Recognition-Gemma/pull/12)

請使用上述標籤／ZIP，不要只下載舊的 v13 Release 或假設 `main` 已合併。ZIP 由該版本的 Git commit 匯出，含完整已追蹤原始碼及 `artifacts/realtime/best_current/` 的 42 類 Transformer v12 模型。

ZIP **不是免安裝執行檔**：不含 Python 環境、MediaPipe 官方 `.task`／WASM、私人影片、憑證、私鑰或現場診斷紀錄。首次安裝需要網路取得 Python 套件與官方資產；已有符合雜湊的資產可離線重用。

不要執行 `setup_mac_auto_trigger.sh`／`run_mac_auto_trigger.sh`：這兩個舊工具只適用另一份含影片的私人 Mac 包，會要求本公開包刻意未附的檔案。請依本頁指令操作。

## 這次更換了什麼

- 膝蓋回位採穩定速度與位置防抖，降低追蹤小幅跳動造成的反覆校準。
- 每句時間上限由 5 秒改成 10 秒，適用所有類別，不為個別句子硬放行。
- 回位後才正常切段、送辨識並更新基準；真正逾時、追蹤遺失不硬送辨識。
- 診斷保持即時更新，顯示校準原因、保持進度及「本句時間／上限」。
- 模型仍是 `knee42-transformer-v12`、42 類；**不用重訓、也不用補資料才能替換 Web**。

## 組員替換流程（建議保留舊版，不直接覆蓋）

1. 記下目前服務的啟動指令、Python 環境、port、憑證及反向代理設定；保留整份舊版資料夾。不要將私鑰或 `.env` 上傳 GitHub。
2. 將 ZIP 解壓到新的 `Knee42-Web-Final-20260915` 資料夾。
3. 在新資料夾使用目前可運行 Web 的 Python 環境；若沒有，依下一節建立環境。
4. 取得並驗證資產、檢查模型，先用 8643 port 測試新版本。
5. 通過下列檢查後，再停止舊服務，把正式啟動目錄改為新資料夾，沿用既有的正式 port／HTTPS／存取控制設定。
6. 清除前端或反向代理舊快取，Chrome 強制重新整理；若有異常，停止新版並以原指令重新啟動保留的舊資料夾。

**前端＋Python 後端＋設定必須一起更新，不能只換 `index.html`。** 主要配套是 `webservice/`、`recognition/`、`configs/auto_trigger_knee_web_live.json` 及完整模型 bundle。若組員已有自訂網站外殼，不要直接蓋掉；將本包當作完整參考版，在其測試分支整合 `/stream` 的前後端契約，再驗收後上線。

## Python 環境

本機驗證環境：Windows、Python 3.12.14、PyTorch 2.13.0+cpu、MediaPipe 0.10.35、NumPy 2.5.0、OpenCV 5.0.0.93。依賴固定在 `requirements-transformer.txt`。已有同一環境時可跳過重新安裝。

Windows PowerShell，新建環境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-transformer.txt
```

以下範例中的 `python`，Windows 新環境請換成 `.\.venv\Scripts\python.exe`，Mac／Linux 換成 `./.venv/bin/python`，或先啟用已確認可用的環境。

Mac／Linux 的新環境：

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-transformer.txt
```

Mac 請使用 Apple Silicon 與符合上述固定套件的作業系統；這次沒有重新在 Mac／Linux 實機驗收，不承諾 Intel Mac 或其他架構可直接安裝。若既有環境可用，優先沿用，不趁換版升級依賴。

## 新資料夾啟動與檢查

所有指令都在解壓後的資料夾根目錄執行。

```bash
python scripts/fetch_mediapipe_models.py
python -c "from recognition.transformer.recognizer import Knee42TransformerRecognizer; r=Knee42TransformerRecognizer('artifacts/realtime/best_current'); assert len(r.labels)==42; print('42-class model integrity OK')"
python -m webservice.server --host 127.0.0.1 --http --port 8643 --bundle artifacts/realtime/best_current --hand-model models/hand_landmarker.task --pose-model models/pose_landmarker.task --vendor-dir webservice/vendor/mediapipe --trigger-config configs/auto_trigger_knee_web_live.json
```

第一行從官方來源抓取並驗證固定版本資產。已有資產時可先將舊版的 `models/` 與 `webservice/vendor/mediapipe/` 複製進新資料夾，再執行驗證；有雜湊差異時讓工具重新下載，不任意替換版本。啟動指令明確指定模型與資產位置，避免舊環境變數仍指向其他 bundle。

在執行伺服器的同一台電腦開啟：

- `http://127.0.0.1:8643/health`：`ok: true`、模型 `knee42-transformer-v12`、`classes: 42`。
- `http://127.0.0.1:8643/?v=web-final-20260915`：相機辨識頁面；不要用 `/replay.html` 取代相機頁。
- `/replay.html` 是額外的影片切段診斷工具，不需要提供私人影片才可使用相機辨識。

手動模式先試一句，再切「自動偵測」，讓肩膀、髖部、膝蓋與雙手入鏡：雙手回膝蓋校準 → 抬手比一句 → 回膝蓋 → 收到结果／基準版本更新。確認「本句時間／上限」顯示 10 秒。再試「我肚子餓」「我聽不懂」及连续兩句；定位品質仍受人物／攝影機視角影響。

正式替換時，**不要把外部網站改成未加密的區網 HTTP**。本機測試指令只供同機使用；既有 Web 的 HTTPS 憑證、反向代理及存取控制需由組員保留。Python 測試伺服器本身沒有登入驗證，不可直接暴露到公網。若由反向代理提供服務，依既有架構轉送所有 API 與 `/vendor/mediapipe/` 資產，不只轉送首頁。

## 驗證結果與尚未宣稱的事

- Python：426 項測試、另 4 subtests 通過；前端 Node：16 項通過。
- 正常、約 8 秒及期限附近回位的合成骨架，經實際模型 HTTP 各得到 1 個結果；這是切段／推論通路測試，不是詞句正確率。
- 使用者回覆「這版蠻好的，可以當最終版」；本版依此凍結為交接基準。不同簽者、其他攝影機與組員線上環境仍需部署後冒煙測試。
- 不把這次調整宣稱為模型準確率提升；模型 gate／資料切分限制維持原有說明。

想重跑測試時，另裝 `pytest` 與 `pytest-subtests`，並使用 Node.js 執行：

```bash
python -m pip install pytest pytest-subtests
python -m pytest tests -q
npm test
```

## 打包來源與回復

維護者以 `git archive` 從 Release tag 的 commit 建立 ZIP，不複製未追蹤的工作目錄。ZIP SHA-256 放在同一個 Release，下載後可用 Windows `Get-FileHash` 或 Mac `shasum -a 256` 比對。

回復方式是重新啟動舊資料夾／舊 commit，並將代理切回舊服務；不要用 `git reset --hard` 清掉組員尚未提交的網站改動。
