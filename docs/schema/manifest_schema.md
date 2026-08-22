# Research Manifest Schema

訓練與驗證所使用的資料清單規格。本檔**只定義欄位**，不含任何實際資料列。

- 檔案格式：CSV，UTF-8（含或不含 BOM 皆可，程式以 `utf-8-sig` 讀取）
- 預期位置：`artifacts/knee42/manifests/research_manifest.csv`
- 一列 = 一支影片 = 一個訓練樣本

---

## 必要欄位

程式實際讀取的欄位只有下列五個。缺任一個都會在訓練啟動時失敗。

| 欄位 | 型別 | 說明 | 讀取位置 |
|---|---|---|---|
| `sample_id` | string | **樣本唯一識別碼**，同時是特徵快取的檔名主體：程式會去找 `{feature_dir}/{sample_id}.npz`。必須全域唯一。 | `Knee42Dataset.__init__`、`cache_path()` |
| `label_id` | string | 類別代碼，必須是 `K42_01` 至 `K42_42` 其中之一。任何其他值都會在建立 label 索引時失敗。 | `Knee42Dataset.__getitem__` |
| `display_text` | string | 該類別的中文句意（例：`你好`）。同一 `label_id` 的所有列必須一致。 | 訓練輸出的 display map |
| `split` | string | `train` 或 `dev`。**比對時會 strip 並轉小寫**，故 `Train`、` dev ` 亦可接受。 | `knee42_policy.validate_research_rows()` |
| `signer_id` | string | 錄影者代碼。**比對時會 strip 並轉大寫**。 | 同上 |

## 切分政策（由程式強制）

`recognition/training/knee42_policy.py` 定義：

```python
TRAIN_SIGNERS = frozenset({"L", "P", "X"})
DEV_SIGNERS   = frozenset({"H"})
TEST_SIGNERS  = frozenset({"J"})
```

`validate_research_rows()` 只接受兩種組合，其餘一律拋出 `LeakageError`：

| `split` | 允許的 `signer_id` |
|---|---|
| `train` | `L`、`P`、`X` |
| `dev` | `H` |

**任何 `split=test` 或 `signer_id=J` 的列進入研究程式路徑都會直接中止。** 這是 fail-closed 設計：不是警告，是拋例外。因此交付給審閱者的 manifest **不應包含任何 J 列**。

`train_dev_only()` 在載入資料前就會呼叫 `validate_research_rows()`，所以這道檢查無法被繞過。

## 選用欄位

下列欄位若存在會被保留（`csv.DictReader` 會讀進來，訓練輸出的逐筆預測檔會帶著它們），但**不影響訓練行為**：

| 欄位 | 用途 |
|---|---|
| `trial_id` | 同一 signer 同一類別的第幾次錄製 |
| `original_file_path` / `relative_path` | 原始影片位置，供追溯 |
| `sha256` | 原始影片雜湊，供完整性稽核 |
| `frame_count`、`fps`、`rotation_metadata_degrees` | 影片屬性，供稽核與統計 |

> **交付前請移除**含有本機絕對路徑、伺服器路徑或個人資訊的選用欄位。

## 範例（欄位示意，非真實資料）

```csv
sample_id,label_id,display_text,split,signer_id,trial_id
K42_01_L_001,K42_01,你好,train,L,1
K42_01_H_001,K42_01,你好,dev,H,1
```

## 一致性檢查

載入 manifest 之後，建議先確認：

1. 每個 `sample_id` 都能在 `features_final/` 找到對應的 `.npz`——`Knee42Dataset` 會在初始化時檢查，缺檔會列出前 5 個並拋 `FileNotFoundError`。
2. `label_id` 全部落在 `K42_01`–`K42_42`。
3. `train` 與 `dev` 兩個 split 都涵蓋全部 42 類；任一類在 `train` 缺席會導致該類永遠學不到，在 `dev` 缺席會使 Macro 指標失真。
4. 沒有任何 J 列。

## 雜湊記錄

每次訓練會把 manifest 的 SHA-256 寫入輸出目錄的 `manifest_sha256.txt`，並記入 round plan。若要重現某一輪的結果，manifest 的雜湊必須相符。
