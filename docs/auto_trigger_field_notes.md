# Auto-trigger 實測紀錄

`configs/auto_trigger_knee_v1.json` 是為**坐姿**簽者校準的:程式碼註解寫明它預期
攝影機看得到膝蓋、靜止時手擱在大腿上。README §6 也把「只針對目前拍攝者、
攝影機位置與距離校準」列為已知限制。

這份文件記錄以 Transformer 路徑做的實測,以及量到的失效模式。

## 量到的現象

### 2026-08-29(a):CLI 即時,1,045 秒,37 段

| 收尾方式 | 段數 | 時長 | top-1 中位數 |
|---|---:|---|---:|
| `timeout_finalize` | **28(76%)** | 全部剛好 12.00 秒 | 0.170 |
| `visible_rest_finalize` | 9(24%) | 中位數 6.17 秒 | 0.190 |

### 2026-08-29(b):網頁自動模式,137 秒,11 段

| 收尾方式 | 段數 |
|---|---:|
| `timeout_finalize` | **10(91%)** |
| `visible_rest_finalize` | 1 —— 而且是**校準後的第一句** |

**決定性的證據是段落之間的間隔:**

```
#3 → #4   0.69s
#4 → #5   0.68s
#5 → #6   0.70s        中位數 0.69s
```

設定檔的 `cooldown_sec` 是 **0.67**。這些間隔就是冷卻時間本身,不是使用者休息的時間。
實際發生的是「撐到 12 秒上限 → 強制切斷 → 冷卻 → 立刻又判定開始比劃」,
**整整 137 秒之中,狀態機一次都沒有回到待命**。

## 機制:為什麼靜止判不到

兩件事在程式碼層面就決定了這個結果。

### 一、靜止基準只校準一次,永不更新

```python
def _update_rest_reference(self, timestamp_sec, analysis) -> None:
    if not self.config.reference_rest_enabled or self._rest_reference_signature is not None:
        return
```

`auto_trigger.py` **沒有任何重新校準的路徑**。開頭那一秒學到的姿勢,要用一整個 session。
姿勢、椅子位置或與鏡頭距離只要漂移到超過
`reference_rest_distance_threshold`(該設定檔收緊為 `0.18`,函式庫預設是 `0.28`),
之後就永遠判不到靜止。

唯一正常收尾的那一段是校準後的第一句,與此完全吻合。

### 二、算靜止特徵需要雙手都被偵測到

```python
def _rest_signature(...):
    left_palm = _hand_center(left_hand)
    right_palm = _hand_center(right_hand)
    if left_palm is None or right_palm is None:
        return None
```

少一隻手就退回以姿態手腕為基礎的備援,但該備援同樣要在校準當下
`explicit_hands_detected == 2` 才會被建立。實測時頁面顯示「偵測到的手 **右**」,
只有單手——光線不足或手離開取景都會造成同樣結果。

### 不是效能問題

實測 `/stream` 的往返時間:一般請求中位數 **12.2 ms**、含辨識 **18.7 ms**,
而瀏覽器每 **400 ms** 才送一次,餘裕三十倍以上,24 次請求無一超時。
`max_segment_sec` 比對的也是瀏覽器打的時間戳,不是伺服器處理時間,
所以延遲不會憑空製造出超時。

低幀率確實會讓靜止判定變難(`end_hold_sec` 0.5 秒 × `end_rest_vote_ratio` 0.8,
在 8.5 fps 下那半秒只有約 4 幀,任何一幀雜訊就破功),
但第 2–8 段都是滿速 30 fps 且照樣超時,所以那不是主因。

## 為什麼這對辨識是致命的

段落會被重取樣成 64 幀。訓練資料的句子長度是 1.1–4.4 秒;
一個 12 秒的段落裡,真正的動作只佔約 11 幀,**其餘 53 幀是靜止畫面**。

同一場測試裡唯一的短段落(1.23 秒 / 37 幀)給出 `謝謝 0.38`,是全場最高的幾個之一;
而一段 12 秒的垃圾段落仍給出 `晚安 0.61`。**模型是有能力的,問題在餵給它的東西。**

## 診斷方式

`POST /stream` 會回報 `rest_distance` 與 `rest_threshold`,頁面狀態列直接顯示:

```
偵測 雙手 · 靜止距離 0.142 / 0.18 ✔ 已回到基準
偵測 僅單手 · 靜止距離 算不出來（需要雙手都被偵測到）
```

這一個數字就能分辨三種情況:

| 觀察 | 病因 | 處理 |
|---|---|---|
| 綠色、< 門檻 | 正常 | — |
| 紅色、持續 > 門檻 | 基準對不上 | 放寬門檻,或加入重新校準 |
| 「算不出來」 | 單手偵測不到 | 光線與取景 |

## 候選修法(依把握程度排序)

1. **`max_segment_sec`: 12.0 → 5.0** —— 與坐站無關。超過 5 秒不可能是這 42 類裡的
   單一句,留著 12 秒只是讓失敗的段落塞滿雜訊。
2. **`reference_rest_distance_threshold`: 0.18 → 0.28** —— 回到函式庫預設值。
   先看實測的 `rest_distance` 落在哪裡再決定。
3. **加入週期性重新校準** —— 治本,但要動 `auto_trigger.py`,
   而該檔被 `packaging/knee42_ivcam/auto_trigger_provenance.json` 的雜湊綁住。
4. `hidden_rest_enabled` → `true`,若手垂身側會離開取景。
5. `end_hold_sec` 0.5 → 0.35、`end_rest_vote_ratio` 0.8 → 0.7,若仍大量超時。

**不要手調就定案。** `recognition/evaluation/eval_auto_trigger_boundaries.py`
能在標註過的影片上做網格搜尋,比憑感覺調可靠;
`auto_trigger_knee_v1.json` 本身就是那樣產生的。新設定請另存新檔。

## 尚未取得的資料

實測至今**沒有任何一次附帶正確答案**(比了哪幾句、什麼順序),
所以只能評估切段行為,**無法評估辨識準確度**。
下次測試請一併記錄比劃清單。
