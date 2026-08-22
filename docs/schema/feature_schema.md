# Feature Cache Schema

`features_final` 特徵快取的格式規格。這是訓練端唯一讀取的資料來源——**訓練過程不會碰原始影片**，影片只在特徵抽取階段使用一次。

- 預期位置：`artifacts/knee42/features_final/`
- 一個樣本 = 一個 `.npz` 檔
- 檔名：`{sample_id}.npz`，其中 `sample_id` 來自 manifest（見 [`manifest_schema.md`](manifest_schema.md)）

---

## 檔案內容

以 `numpy.savez` 儲存，**必須包含三個鍵**：

| 鍵 | 型別 | 形狀 | 說明 |
|---|---|---|---|
| `cache_version` | string scalar | — | 必須**逐字**等於 `knee42_features_upright_v2`，否則 `load_cache()` 拋 `ValueError` |
| `values` | float32 | `[N, 219]` | 每幀 219 個座標值 |
| `mask` | bool | `[N, 219]` | 每個座標是否為實際觀測值 |

`N` 為該樣本的特徵幀數，**不固定**，但必須 `> 0`。取樣到 64 幀是在訓練時才做（見下文），不在快取階段。

讀取時使用 `allow_pickle=False`，因此不可存入 Python 物件。

## 219 維的組成

順序固定，不可更動：

```text
索引   0 –  92 : Pose 31 點 × 3   (MediaPipe Pose 33 點移除索引 25、26)
索引  93 – 155 : 左手 21 點 × 3
索引 156 – 218 : 右手 21 點 × 3
```

- 每點為 `(x, y, z)` 三個連續值。
- Pose 保留索引：`tuple(i for i in range(33) if i not in (25, 26))`，即移除左右膝。
- 左右手依 MediaPipe 的 **handedness label** 指派，**不是**依偵測陣列順序。
- 座標已套用**肩寬相對正規化**：以雙肩中點為原點、雙肩 2D 距離為尺度。

## 缺失值的表示（關鍵）

未偵測到的關鍵點，其座標**必須為非有限值（`NaN`）**，對應的 `mask` 為 `False`。

`load_cache()` 有一道 fail-closed 檢查：

```python
if np.any(np.isfinite(values) & ~mask):
    raise ValueError(f"cache mask/value mismatch: {path}")
```

意思是：**任何有限的值都必須被 mask 標記為 True。** 換句話說，不可以先把缺失填成 0 再把 mask 設 False——那會直接被擋下來。

理由：若缺失填 0，該值與「關節確實位於原點」在數值上無法區分，模型會把「看不到手」學成一個特定位置。

## 額外的形狀檢查

`load_cache()` 另外強制：

- `values.ndim == 2`
- `values.shape == mask.shape`
- `values.shape[0] > 0`

## 訓練時如何被使用

快取本身不做取樣與標準化，兩者都發生在 `Knee42Dataset.__getitem__`：

```python
values, mask = load_cache(cache_path(feature_dir, row["sample_id"]))
values, mask = select_fixed_frames(values, mask, sequence_length)   # 取 64 幀
standardized = (values - mean) / std                                # train-only 標準化
standardized = np.where(mask, standardized, 0.0)                    # 中性填補
features = np.concatenate([standardized, mask.astype(np.float32)], axis=1)   # → [64, 438]
```

**取樣**（`select_fixed_frames`）：

```python
indices = np.rint(np.linspace(0, len(values) - 1, sequence_length)).astype(np.int64)
```

在整段的頭到尾之間取 64 個等距位置再取最近幀。`N > 64` 時跳著取，`N < 64` 時索引會重複（等同最近鄰上取樣）。

**中性填補**：缺失位置在標準化**之後**填 `0.0`。標準化空間的 0 即訓練集平均，是最沒有資訊量的值；`mask` 則保留「這格是補的」這項事實。填補與遮罩是一組配套，缺一不可。

**最終輸入**：`[64, 219 標準化值] + [64, 219 遮罩]` 串接為 `[64, 438]`。

## 標準化參數如何產生

`fit_standardizer()` **只使用 `split=train` 的列**，並且只統計**被觀測到的**座標：

```python
observed     = np.where(mask, values, 0.0)
count[d]     = mask[:, d].sum()          # 該維度實際被觀測的次數
mean[d]      = total[d] / count[d]
variance[d]  = max(squared[d]/count[d] - mean[d]**2, 1e-6)
```

- 缺失位置**不參與**平均與變異數計算，不會把 NaN 或填補值算進統計。
- 變異數有下限 `1e-6`，避免除以 0。
- 若任一維度在整個訓練集都未被觀測到，會拋 `ValueError` 並列出該維度索引。

輸出存成 `standardizer_train_only.npz`，形狀為兩個 `[219]` 陣列。**Dev 與 Test 的分布完全不參與**，這是避免資訊洩漏到前處理階段的關鍵。

## 完整性驗證

抽取完成後應產生 feature ledger，記錄每個 `.npz` 的 SHA-256 與整體聚合雜湊。訓練輸出目錄會寫入 `feature_ledger_sha256.txt`，用於重現特定一輪的結果。

## 自我檢查

拿到一份特徵快取後，建議先跑一次：

```python
import numpy as np, glob, os

paths = glob.glob("artifacts/knee42/features_final/*.npz")
print("樣本數:", len(paths))
bad = []
for p in paths:
    with np.load(p, allow_pickle=False) as z:
        if str(z["cache_version"].item()) != "knee42_features_upright_v2":
            bad.append((p, "cache_version")); continue
        v, m = z["values"], z["mask"]
        if v.ndim != 2 or v.shape[1] != 219 or v.shape != m.shape or v.shape[0] == 0:
            bad.append((p, f"shape {v.shape}/{m.shape}")); continue
        if np.any(np.isfinite(v) & ~m):
            bad.append((p, "mask/value mismatch"))
print("異常:", bad[:5], f"（共 {len(bad)} 筆）")
```

全部通過即可直接進入訓練，不需要任何影片檔。
