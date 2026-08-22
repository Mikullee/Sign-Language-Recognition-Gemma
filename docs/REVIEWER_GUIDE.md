# 審閱者指引

給指導教授與外部審閱者。這份文件的用途是把「看懂 → 核對模型 → 自己重訓」排成一條可依序執行的路徑。

專案主文件為 [README.md](../README.md)；本檔只負責排序與補上**不公開資料的取得與放置方式**。

---

## 第一部分：看懂（約 20 分鐘）

依序閱讀 README 的三節：

| 讀什麼 | 得到什麼 |
|---|---|
| [§1 這是什麼](../README.md#1-這是什麼) | 系統的範圍界定——它做 42 類固定句型分類，不做連續手語翻譯，辨識核心與 Gemma 無關 |
| [§3.1 完整處理路徑](../README.md#31-完整處理路徑) | 一張流程圖看完整條管線。關鍵是分岔與再合流：MediaPipe 每幀只跑一次，輸出兩種表徵，切段那一路決定辨識那一路看得到哪些畫面 |
| [§2 目前辨識效果](../README.md#2-目前辨識效果) | 現況與其限制 |

**目前進度一句話**：模型 gate 為 `PROVISIONAL`，Dev Macro Top-1 76.35%、一次性 J 測試 62.34%；軟體管線、封裝與可重現性證據皆已通過，未達 READY 的原因是 8 類 J 零準確率、三種子變異 3.04 pp、以及實機硬體驗收未完成。

若只想看「哪些類別能用、哪些不能」，直接看 §2.3 的四組分類與 [`evaluation/live_check_42.csv`](evaluation/live_check_42.csv)。

**兩件請特別留意的事**：

1. **Dev 與 J 的性質不同。** Dev 被用來挑選回合、種子與 checkpoint，帶有樂觀偏誤；J 是唯一未參與任何選擇的估計值。兩者出現落差屬預期。
2. **J 的一次性額度已消耗。** 任何後續重訓的模型**只能報告 Dev 指標**，不得再對 J 評估。若需要新的 signer-independent 數字，必須另建 held-out 測試集。

---

## 第二部分：核對公開模型（約 10 分鐘）

從 GitHub Release `v1.0.0-v13` 下載：

- `knee42-model-v11.zip`：權重、label map、standardizer、feature config 與中文顯示對照
- `SHA256SUMS.txt`：Release 附件雜湊

先核對附件完整性：

```powershell
Get-FileHash .\knee42-model-v11.zip -Algorithm SHA256
```

本次公開範圍是**模型與訓練／推論方法**，不包含 Windows 可攜包、MediaPipe `.task` 或人物影片。模型 bundle 不是獨立可執行應用程式；如需即時硬體展示，請聯絡專案維護者另行取得必要資產與說明。

---

## 第三部分：自己重訓

### 需要另外取得的兩份資料

repository **不含**任何影片或逐筆特徵快取（原因見 [§8.1](../README.md#81-不公開的內容)：訓練資料源自專案成員錄製的手語影像）。重訓需要下列兩項，由專案維護者透過私人管道另行提供：

| 項目 | 放置位置 | 內容 |
|---|---|---|
| Research manifest | `artifacts/knee42/manifests/research_manifest.csv` | Train/Dev 樣本清單，**不含任何 J 列** |
| 特徵快取 | `artifacts/knee42/features_final/` | 每樣本一個 `.npz`，內含 219 座標值與 219 遮罩的時序 |

**放進這兩個位置之後就不需要任何其他設定。** 訓練端完全不讀原始影片——影片只在特徵抽取階段使用一次，該階段已完成。

規格見 [`schema/manifest_schema.md`](schema/manifest_schema.md) 與 [`schema/feature_schema.md`](schema/feature_schema.md)。放置完成後，執行 `scripts/verify_knee42_features_final.py` 檢查完整性。

### 環境

重現訓練需使用相容的 **Linux／CUDA** 主機：

```bash
conda create -n knee42 python=3.10 pip -y
conda activate knee42
python -m pip install -r requirements.lock.txt
```

`requirements.lock.txt` 是目前發布模型的 Linux／CUDA 精確套件版本快照，不適用於 Windows 推論環境。`environment.yml` 僅供快速建立未完全鎖版的基礎環境；如使用該檔，環境名稱為 `slr_runtime`，仍須再安裝 `requirements.lock.txt` 才能對齊訓練版本。`requirements-windows.txt` 僅記錄來源層級推論所需套件；公開 Release 不含完整即時執行資產。

### 重現目前發布的模型

`configs/knee42/round1_config.json` 即 MODEL v11 所用的設定（64 幀、coordinate jitter、hidden 128、dropout 0.45、weight decay 1e-4）。

```python
import csv, json
from pathlib import Path
from recognition.training.knee42_devonly import DevOnlyConfig, train_dev_only

rows = list(csv.DictReader(open(
    "artifacts/knee42/manifests/research_manifest.csv",
    encoding="utf-8-sig", newline="")))
cfg = DevOnlyConfig(**json.load(open("configs/knee42/round1_config.json", encoding="utf-8")))

train_dev_only(
    rows=rows, config=cfg,
    split_hash="<split_sha256>",
    manifest_hash="<manifest_sha256>",
    feature_dir=Path("artifacts/knee42/features_final"),
    out_dir=Path("artifacts/knee42/iterations/reproduce/seed44"),
    seed=44,
)
```

seed 44 應可得到接近 **Dev Macro Top-1 76.35%** 的結果。三種子參考值為 75.39 / 69.48 / 76.35（mean 73.74、population std 3.04 pp）——**這個變異幅度本身就是目前的已知風險之一**，單一種子的結果不足以下結論。

### 自己做實驗

專案採 Round 制度，每輪只變動**單一因素**，並以 JSON 計畫書記錄假設與成功條件。既有的 11 輪紀錄見 [§5.5](../README.md#55-研究流程與-round-制度)。

一個已知的開放問題：**序列長度**。Round 3（96 幀，80.95%）與 Round 10（128 幀，83.02%）在 Dev 上都高於選定的 64 幀（76.35%），但三者都只跑了 seed 44，而 Round 1 自身的三種子全距達 6.87 pp，因此該領先尚未被證實。詳細討論與建議的驗證設計見 [§3.5](../README.md#35-為什麼是-64)。

### 不可跨越的界線

1. **不得在 J 上評估任何模型。** 一次性封印已消耗。`recognition/training/knee42_policy.py` 的 `validate_research_rows()` 會在任何 `split=test` 或 `signer_id=J` 的列進入研究路徑時拋出 `LeakageError`，這道檢查無法繞過。
2. **不得以 J 的結果決定排除哪些訓練資料**、挑選實驗方向或調整超參數。
3. **不覆寫既有的時間戳證據目錄**，新輪次寫入新目錄。

---

## 資料使用範圍

另行提供的 manifest 與特徵快取，僅供本專題之審閱與結果重現使用，請勿轉散布。特徵快取為骨架座標序列，不含任何影像，無法還原為影片；但仍屬專案成員提供之資料。

模型權重採 CC BY-NC 4.0，允許研究、教學與非商業測試，須標註來源。程式碼採 MIT。

---

## 有問題時

- 環境或執行問題：GitHub Issue（請勿上傳含人物影像的原始影片）
- 資料取得、訓練細節：聯絡專案維護者
