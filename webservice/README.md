# 瀏覽器測試服務

讓別人不用安裝任何東西就能試模型的網頁。三種輸入:攝影機、上傳影片、貼影片連結。

```bash
python -m webservice.server --port 8642
```

開 `https://<主機>:8642`。**必須是 https**——瀏覽器只在安全來源提供 `getUserMedia`,
純 http 的頁面連要求攝影機權限的機會都沒有。

## 啟動前需要的東西

| 需求 | 說明 |
|---|---|
| 模型 bundle | 預設讀 `artifacts/realtime/best_current/`,repo 內附 |
| MediaPipe `.task` | 放在 `models/`(見 [`models/README.md`](../models/README.md)) |
| MediaPipe 網頁資產 | 攝影機模式需要,見下方 |
| TLS 憑證 | 第一次啟動會用 `openssl` 自簽;沒有 openssl 就用 `--certfile` / `--keyfile` 自備 |

### MediaPipe 網頁資產

攝影機模式的 MediaPipe **在瀏覽器裡跑**,需要 `vision_bundle.mjs`、wasm 與 `.task`。
本 repository 不重新散布這些檔案(見 README §4.3 的公開範圍),請自行取得後放進一個目錄:

```
<vendor-dir>/
├── vision_bundle.mjs
├── wasm/
│   ├── vision_wasm_internal.js
│   ├── vision_wasm_internal.wasm
│   ├── vision_wasm_nosimd_internal.js
│   └── vision_wasm_nosimd_internal.wasm
├── hand_landmarker.task
└── pose_landmarker_lite.task
```

然後 `--vendor-dir <path>`,或設環境變數 `SLR_WEB_VENDOR_DIR`。
**上傳影片與貼連結兩種模式不需要這些**,只有攝影機模式要。

## 三種模式

### 攝影機

按住空白鍵錄 1–3 秒,放開出結果。MediaPipe 在瀏覽器端執行,
**只有骨架座標會 POST 到 `/predict`,畫面不會離開使用者的電腦**。

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
| POST | `/predict` | `{frames:[{timestamp,pose,hands}]}` → 切段結果 + top-5 |
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
