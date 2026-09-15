# Web 膝蓋回位自動切段候選版

這一版只改善「何時開始、何時結束」，沿用 repository 內的 42 類 Transformer v12。
沒有重訓分類模型，也沒有修改 219 維辨識特徵、手部左右槽位或模型檔案。
不要把這次的切段測試當成新的辨識準確率。
本輪實際數據見 [2026-09-15 驗證摘要](web_knee_candidate_results_20260915.md)。

## 直接測試

在本分支的 repository 根目錄，使用已安裝 `requirements-transformer.txt` 的 Python：

```bash
python scripts/fetch_mediapipe_models.py --browser
python -m webservice.server --host 127.0.0.1 --http --port 8642
```

已有 MediaPipe 資產時，第一行會先驗證雜湊；不需要重新下載全部資料。
開啟 `http://127.0.0.1:8642/` 測模型；開啟 `http://127.0.0.1:8642/replay.html` 測自動切段影片。
本機 HTTP 只綁 loopback，沒有開放區網。停止服務按 Ctrl-C。

即時測試先用預設「手動」模式確認能出結果，再切換自動模式：

1. 坐好，讓肩膀、髖部、膝蓋和雙手腕完整入鏡。
2. 雙手放回膝蓋／下段大腿，穩定約 1 秒完成第一次校準。
3. 抬手比手語。停在胸前不會當成結束。
4. 雙手回到膝蓋，持續約 0.5 秒確認結束；切片邊界回到最初確認的回位時間，不含整段等待時間。
5. 冷卻期間持續回位，更新一次基準，等待下一句。每次有效回位只更新一次。

畫面會顯示等待膝蓋可見、初始校準、等待回位、手腕備援、動作速度和基準版本。
「靜止距離很小」本身不代表合格回位；綠色確認依據完整回位判定。
超過 5 秒沒有正常結束，該段列為失敗、不送模型猜答案，請雙手回膝蓋重來。

膝蓋短暫被遮住時，僅在既有基準可信、身體未明顯移位且雙腕可見下，允許最多 1 秒備援。
備援能協助結束，但不能建立或更新基準。缺手腕、畫面中斷或整個人換位置時不湊假資料。

## 一鍵批次評估

初次安裝評估用的 Node.js 20+ 依賴（即時辨識本身不需要 Node.js）：

```bash
npm ci
npx playwright install chromium
```

將私人影片放在本機資料夾，標註保留 `video_path,expected_label,start_sec,end_sec,notes`。
`--video-root` 以檔名尋找影片，不需要把本機絕對路徑寫進公開 CSV。

```bash
python -m recognition.evaluation.web_trigger_replay --video-root ./my-private-videos
```

若 8642 沒有服務，指令會自行啟動本機服務，評估後只關閉它自己啟動的程序。
已有服務則共用，不關閉使用者的服務。若不下載 Chromium，可以使用已安裝的 Chrome：

```powershell
$env:PLAYWRIGHT_CHANNEL = "chrome"
python -m recognition.evaluation.web_trigger_replay --video-root ./my-private-videos
```

macOS/Linux 對應環境變數寫法為 `PLAYWRIGHT_CHANNEL=chrome python -m recognition.evaluation.web_trigger_replay --video-root ./my-private-videos`。
需要驗證 HTTPS 自簽憑證時建議改用上述本機 HTTP 入口。

常用選項：

- `--annotations ./annotations.csv`：指定人工標註。
- `--original-only --skip-tuning`：只跑原片快速檢查，不是完整擴增驗證。
- `--reuse-cache`：只使用完整且雜湊吻合的既有瀏覽器快取；缺檔即報錯。
- `--out-dir ./results/my-run`：指定輸出位置，建議維持在忽略的 results 目錄。

預設每原片 20 組擾動：0.75／1／1.25 倍速、10／15／17.5／30 FPS、位置、大小、時間抖動、手部遺失及膝蓋遮擋。
圖像變換先在 canvas 執行再重新抽骨架；明確標為手部／可見度遺失的案例則是骨架級故障注入。
不產生原片不存在的膝蓋座標。影片不離開本機瀏覽器；回放只送骨架到 loopback 服務。
私人影片回放頁只允許 localhost／127.0.0.1／IPv6 loopback；在區網或公開網址會停用檔案選擇與回放。

輸出位於 `results/web_knee_candidate/`：

- `report.md`、`summary.json`：新舊比較、起訖誤差、漏切、多切、提前截斷、逾時、缺資料與 EOF。
- `candidate_config.json`：候選設定；程式不會自動覆蓋正式設定。
- `private_overlays/`：原片上疊加 GT／舊版／新版邊界，僅供本機目視核對。
- `private_cache/`、`private_traces/`、`private_capture_manifest.json`：含骨架或私人路徑，不應公開。

請勿把整個 results 資料夾上傳 GitHub。這次公開的只有程式碼、測試與不含人物影像的摘要。

## 結果如何解讀

目前三支舊影片沒有清楚拍到膝蓋，不能拿來驗收「手回膝蓋」的成功率。
它們只用於一般邊界診斷與缺膝蓋時的拒絕測試；因此本輪不會宣布膝蓋設定已完成資料驅動調優。
這類影片的新版邊界通過欄標為 N/A，不把缺膝蓋時正確等待誤報為切段成功或失敗。
正式流程另以合成骨架測試：兩句連續輸入、重新定位、短暫遮擋、位置／速度／FPS／時間抖動。
合成通過只證明程式對這些情境符合規格，不代表對不同真人都有效。

同一原片及其全部擴增以來源 SHA 分在同一組，不得跨調參／驗證兩側。
一般邊界診斷最多搜尋 16 組舊式設定，做三原片輪流保留驗證，**結果不會裝進膝蓋正式模式**。
要進行正式膝蓋設定選擇，須人工確認影片有完整膝蓋協定，再新增 `knee_protocol=yes` 欄位；
至少需要 3 個不同原片 SHA，才會進行分組驗證及選擇。不同原片仍不等於不同簽者。
最終用全部合格資料選出的設定，其比較圖為選擇後資料內結果；獨立來源結果另列在 grouped_knee_folds。

本次沿用工程預設，不以不相容站姿影片強行選出「最佳膝蓋模型」。
仍需真實膝蓋入鏡和不同人物的現場驗收；有此類資料之前，跨人切段效果是未知的。
這是協定式「每句雙手回膝蓋」，不是自然連續手語的語言學斷句器。

## 版本、API 與相容性

- 現行 Web 設定：`configs/auto_trigger_knee_web_live.json`。速度門檻改為肩寬／秒，只影響切段。
- 先前 Web 設定：`configs/auto_trigger_knee_web_previous.json`，供診斷／回退使用，仍有舊流程限制。
- `recognition/realtime/auto_trigger.py` 保持歷史雜湊不變；舊 v13 ZIP 沒有被重寫。
- 精確舊 Web 引擎快照：`recognition/evaluation/baselines/web_trigger_0ae3ac0.py`，只用於比較，不能當正式入口。
- `POST /stream` 的 pose 多一個 `visibility[33]`；缺可見度的舊客戶端在嚴格膝蓋模式不會完成校準，請重新整理網頁。
- 219 維辨識仍只用 xyz；可見度只進切段。新增回傳 `events`、`failure_count`、`last_message`、`motion_score`、`knees_visible`、`rest_candidate`。
- `events[].accepted=false` 表示逾時、資料中斷或過短等失敗；`results` 僅含合格並送模型的片段。
- 重複／過期時間戳被丟棄並計數；不改寫成合成時間。超過 0.25 秒的觀測斷流不能補算休息。
- `eof=true` 只標記完整／未完成 EOF，不複製最後一幀湊足確認時間。
- 相機可使用 GPU，回放固定 CPU；二者共用 VIDEO 模式、模型檔案和序列化，但不宣稱不同運算後端產生逐位元相同的骨架。
- 瀏覽器回放會比對實際服務提供的 JS／WASM／模型 SHA，並記錄瀏覽器版本；不同來源的舊 IMAGE 快取不混用。

驗證：`python -m pytest tests -q`；Node.js 測試：`npm test`。
本輪在 Windows 驗證；提供跨平台指令，但尚未在真實 Mac 上完成實機驗證。

## 文獻採用範圍

- [Few-shot Skeleton-based Temporal Action Segmentation](https://arxiv.org/abs/2207.09925)：借鑑受控骨架／時序擴增；未重現其網路或 CTC 訓練。
- [Sign Language Segmentation with Temporal Convolutional Networks](https://arxiv.org/abs/2011.12986)：把邊界品質與分類正確率分開評估。
- [MS-TCN](https://arxiv.org/abs/1903.01945)：借鑑時間連續性、避免過度切分的原則；本版不是 MS-TCN 模型。

沒有把上述論文的準確率套用到 Knee42，也沒有宣稱擴增可以代替新的簽者。
