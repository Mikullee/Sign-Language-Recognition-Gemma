# 瀏覽器測試服務

目前交接使用 `web-final-20260915`；下載、安裝與替換既有 Web 請先看 [最終換版說明](../docs/web_final_handoff_20260915.md)。歷史候選版操作與批次評估見 [候選版說明](../docs/web_knee_candidate.md)。
`/replay.html` 使用與相機相同的 Web VIDEO 模式和切段控制器；下方舊的上傳影片分析是另一種分析功能，不能用來驗收新版自動觸發。

讓別人不用安裝任何東西就能試模型的網頁。三種輸入:攝影機、上傳影片、貼影片連結。

```bash
python -m webservice.server --port 8642
```

開 `https://<主機>:8642`。瀏覽器只在安全來源提供 `getUserMedia`。若只在同一台
電腦測試，可免憑證啟動：

```bash
python -m webservice.server --host 127.0.0.1 --port 8642 --http
```

再開 `http://127.0.0.1:8642`；localhost 是瀏覽器認可的安全來源。`--http` 會拒絕
`0.0.0.0`、區網 IP 與網域名稱，不能用它把未加密服務開給其他電腦。

## 啟動前需要的東西

| 需求 | 說明 |
|---|---|
| 模型 bundle | 預設讀 `artifacts/realtime/best_current/`,repo 內附 |
| MediaPipe `.task` | 放在 `models/`(見 [`models/README.md`](../models/README.md)) |
| MediaPipe 網頁資產 | 攝影機模式需要,見下方 |
| TLS 憑證 | 區網使用時需要；第一次啟動會用 `openssl` 自簽，沒有 openssl 就用 `--certfile` / `--keyfile` 自備；本機可用 `--http` |

### MediaPipe 網頁資產

攝影機模式的 MediaPipe **在瀏覽器裡跑**,需要 Tasks Vision 的 WebAssembly 建置。
本 repository 不散布這些檔案(見
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)),但抓取腳本會處理:

```bash
python scripts/fetch_mediapipe_models.py
```

它會取得並逐檔核對 SHA-256,放進 `webservice/vendor/mediapipe/`:

```
vendor/mediapipe/
├── vision_bundle.mjs
├── wasm/vision_wasm_internal.{js,wasm}
├── wasm/vision_wasm_nosimd_internal.{js,wasm}
├── hand_landmarker.task
└── pose_landmarker_lite.task
```

版本固定在 `@mediapipe/tasks-vision` **0.10.35**,與 Python 端的 mediapipe 同版。
要放在別處就用 `--vendor-dir`,或設環境變數 `SLR_WEB_VENDOR_DIR`。

**上傳影片模式不需要這些**,只有攝影機模式要。

## 三種模式

### 攝影機

右上角可切換兩種切段方式:

| 模式 | 操作 |
|---|---|
| **手動**(預設) | 按住空白鍵錄 1–3 秒,放開出結果 |
| **自動偵測** | 膝蓋完整入鏡，雙手回膝蓋穩定約 1 秒校準；每句結束後回膝蓋重新定位 |

MediaPipe 在瀏覽器端執行,**只有骨架座標會 POST 給伺服器,畫面不會離開使用者的電腦**。

自動模式的**起訖判定在伺服器上**,走 `POST /stream`:頁面每 400 毫秒把累積的
landmark 送上去。伺服器以封存的 `recognition.realtime.auto_trigger` 為基底，透過
`recognition.transformer.live_trigger` 加入目前 Web 專用的重新校準與 Pose 手腕備援；
段落結束時連同 top-5 一起回傳。舊 V13 的受雜湊鎖定程式碼不會被改寫。

**狀態機不在瀏覽器重寫。** 回放與即時串流共用 Python 控制器；頁面只負責抽骨架、送資料和顯示。
目前設定已由使用者在其 Chrome／攝影機環境試用並接受為最終版基準；不同人物與其他部署環境仍需驗收，不能宣稱已完成跨人量化驗證。

伺服器為每個分頁維持一份狀態機(它要校準靜止基準、還要保留 pre-roll 緩衝,
不能每次請求重建),閒置五分鐘回收。

頁面送的是**原始、未翻轉**的 Tasks API 輸出,handedness 原樣保留——
這就是訓練特徵快取的慣例。**不要在前端做任何水平翻轉或左右交換**,
那會讓推論落在跟訓練不同的分布上。

### 上傳影片

≤ 200 MB、≤ 180 秒。伺服器逐幀追蹤、依手腕動能切段,每段給 top-3,
另外附一個整段的結果(單詞影片常常切不出段,整段結果才是答案)。
一次只跑一件,其餘排隊——MediaPipe tracker 不能共用。

### 貼連結

需 `--allow-url-fetch` 且系統有 `yt-dlp`。預設關閉,因為它會對外連線。

## API

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/` | 測試頁 |
| GET | `/health` | 模型 id、類別數、各項上限 |
| GET | `/labels` | 42 類清單與中文對照 |
| POST | `/predict` | `{frames:[{timestamp,pose,hands}]}` → 切段結果 + top-5(手動模式) |
| POST | `/stream` | `{session,frames,reset?}` → `{state,calibrated,results}`(自動模式) |
| POST | `/analyze_upload` | multipart,欄位名 `file` → `{job_id}` |
| POST | `/analyze_url` | `{"url": "…"}` → `{job_id}` |
| GET | `/job/<id>` | `{state, phase, done, total, result, error}` |

```bash
curl -k https://127.0.0.1:8642/health
curl -k -X POST -F "file=@clip.mp4" https://127.0.0.1:8642/analyze_upload
curl -k https://127.0.0.1:8642/job/<job_id>
```

## 結果怎麼看

- 42 類是**封閉集合**,遇到詞表外的手語一定會亂答,不會說「不知道」。
- 回傳的 `left_shoulder_x` 應該接近 **+0.5**。這是對**正規化是否正確**的健檢——
  **不是**鏡像偵測。姿態模型會依身體外觀判斷解剖左右,畫面鏡像與否這個值都會是正的。
- 影片真的是自拍鏡像的話,要在**送進 MediaPipe 之前**轉正
  (`analyze_video(..., selfie_flip=True)`),事後修座標不等價。

## ⚠️ 沒有身分驗證

任何連得到這個 port 的人都能使用。這是設計為**內部測試工具**,
請綁在受信任的網段,或放在會做驗證的反向代理後面。不要直接開到公網。

上傳的影片寫到系統暫存目錄,處理完立即刪除;工作紀錄保留一小時後清除。

## 已知限制

- 影片工作一次只跑一件,其餘排隊。
- 只提供 42 類詞句模型。教授交接包裡另有 215 詞模型,但本 repository 的
  bundle 合約只驗 42 類,尚未納入。
- 自簽憑證每次換瀏覽器都會跳一次警告,屬正常現象。
