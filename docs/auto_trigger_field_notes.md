# Auto-trigger 實測紀錄

`configs/auto_trigger_knee_v1.json` 是為**坐姿**簽者校準的:程式碼註解寫明它預期
攝影機看得到膝蓋、靜止時手擱在大腿上(`knee_geometry_enabled`、
`knee_min_thigh_progress_ratio` 等參數都是為此存在)。README §6 也把
「只針對目前拍攝者、攝影機位置與距離校準」列為已知限制。

這份文件記錄第一次以 Transformer 路徑做的實測,以及量到的失效模式。

## 2026-08-29:1,045 秒即時測試

37 個段落:

| 收尾方式 | 段數 | 時長 | top-1 中位數 |
|---|---:|---|---:|
| `timeout_finalize` | **28(76%)** | **全部剛好 12.00 秒** | 0.170 |
| `visible_rest_finalize` | 9(24%) | 中位數 6.17 秒 | 0.190 |

**四分之三的段落不是「偵測到結束」,是撐到 `max_segment_sec` 上限被硬切。**

### 為什麼這對辨識是致命的

段落會被重取樣成 64 幀。訓練資料的句子長度是 1.1–4.4 秒;
一個 12 秒的段落裡,真正的動作只佔約 11 幀,**其餘 53 幀是靜止畫面**。
模型看到的幾乎全是雜訊,信心自然落在 0.06–0.2。

同一場測試裡唯一一個乾淨的短段落——#36,1.23 秒 / 37 幀——
給出 `謝謝 0.38`,是全場最高的幾個之一。這是「模型沒問題、切段有問題」的直接證據。

### 狀態機的擺盪

失效的段落普遍出現這種來回:

```
SIGNING_ACTIVE → END_CONFIRM → SIGNING_ACTIVE → END_CONFIRM → ...
```

動作能量有掉到門檻以下,但靜止判定始終不成立,於是反覆退回。

## 候選原因(尚未逐一驗證)

**攝影機看不到膝蓋。** 筆電或桌上型鏡頭通常只拍到頭與上半身。
`knee_geometry_enabled: true` 想用的幾何資訊不存在時,只能退回靜止基準比對,
而該門檻被收緊到 `0.18`(函式庫預設是 `0.28`)。

**手垂在身側離開取景範圍。** 靜止判定裡:

```python
hidden_rest_blank = (
    config.hidden_rest_enabled          # 目前是 False
    and torso_valid
    and explicit_hands_detected == 0
    and torso_motion <= config.blank_motion_threshold
)
```

關閉此項等於宣告「看不到手就不算靜止」。坐姿時手在大腿上仍被偵測到,
所以不成問題;手離開畫面時段落就永遠等不到結束。

**光線不足。** 現場回報懷疑環境太暗。MediaPipe 追不到手時,
`explicit_hands_detected` 同樣歸零,效果與上一項相同。

## 下一步

先排除環境因素(光線、取景是否含膝蓋),再調參數。目前最有把握的一項是:

- **`max_segment_sec`: 12.0 → 5.0** —— 超過 5 秒的段落不可能是這 42 類裡的單一句,
  留著 12 秒只是讓失敗的段落塞滿雜訊。這一項與坐站無關。

其餘視實際情況:

| 情況 | 該調的參數 |
|---|---|
| 鏡頭看不到膝蓋 | `knee_geometry_enabled` → `false`、`reference_rest_distance_threshold` → `0.28` |
| 手垂身側離開畫面 | `hidden_rest_enabled` → `true` |
| 仍大量超時 | `end_hold_sec` 0.5 → 0.35、`end_rest_vote_ratio` 0.8 → 0.7 |

**不要手調就定案。** `recognition/evaluation/eval_auto_trigger_boundaries.py`
能在標註過的影片上做網格搜尋,比憑感覺調可靠;
`configs/auto_trigger_knee_v1.json` 本身就是那樣產生的。

新設定請另存新檔,不要覆蓋 `auto_trigger_knee_v1.json`——
它被 `packaging/knee42_ivcam/auto_trigger_provenance.json` 的雜湊綁住。
