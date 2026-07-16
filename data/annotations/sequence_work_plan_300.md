# Sequence 錄影與資料整理任務（300 支）

## 目標

本輪先建立 `300` 支 sequence 影片資料，作為後續連續手語辨識與 `HST-SLR` 方向的訓練資料。

- 總模板數：`12` 組
- 每組影片數：`25` 支
- 每支影片長度：`3~4` 個 gloss
- 總影片數：`12 x 25 = 300`

---

## 分工

### 1. 錄影組
- 林芯羽（A）
- 廖嘉儀（C）

### 2. 審片 + 檔名整理組
- 謝承翰

### 3. 標註組
- 梁曼璇
- 林姵妏

### 4. Gloss Description + 文件整理組
- 徐姿琪

---

## 影片命名格式

請統一使用：

```text
SEQ編號_錄影者代號_次數.mp4
```

範例：

```text
SEQ01_A_01.mp4
SEQ01_C_01.mp4
SEQ12_A_12.mp4
SEQ12_C_13.mp4
```

錄影者代號：

- `A`：林芯羽
- `C`：廖嘉儀

---

## 12 組 Sequence 模板

1. `今天 大陸 冷氣團`
2. `冷氣團 北部 東北部`
3. `北部 東北部 晚 冷`
4. `冷 越來越 其他 地區`
5. `地區 早晚 大家`
6. `大家 晚 歸`
7. `歸 記得 保暖`
8. `記得 保暖 做好 要`
9. `今天 冷氣團 北部 晚`
10. `東北部 冷 越來越`
11. `其他 地區 早晚 大家`
12. `晚 歸 記得 要`

---

## 錄影數量分配

- `SEQ01 ~ SEQ06`
  - 林芯羽各錄 `13` 支
  - 廖嘉儀各錄 `12` 支

- `SEQ07 ~ SEQ12`
  - 林芯羽各錄 `12` 支
  - 廖嘉儀各錄 `13` 支

### 各自總量

- 林芯羽：`150` 支
- 廖嘉儀：`150` 支
- 總共：`300` 支

---

## 各組詳細工作

### 1. 錄影組

負責內容：

- 依照 `12` 組 template 錄完全部 `300` 支影片
- 確保 gloss 順序與模板一致
- 同一模板可錄正常、稍快、稍慢版本
- 若審片組回報不合格，需補錄

錄影規則：

- 上半身完整入鏡
- 手不要出框
- 鏡頭固定
- 背景盡量單純
- 光線穩定
- 每支影片開頭和結尾各留 `0.5 ~ 1` 秒空白
- gloss 之間銜接自然，不要太快
- 不要邊錄邊講話

特別注意：

- `其他` 只限定為「其他地區」這個語境
- 不能把不確定的動作或模糊段落都錄成 `其他`

錄影組要用到的檔案：

- [sequence_templates_12.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_templates_12.csv)
- [sequence_recording_manifest_300.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_recording_manifest_300.csv)

---

### 2. 審片 + 檔名整理組

負責內容：

- 檢查錄好的影片是否符合規格
- 確認 gloss 順序是否正確
- 確認是否需要重錄
- 核對檔名是否符合命名格式
- 整理影片檔案到正確資料夾

審片時要檢查：

- 人有沒有完整入鏡
- 手有沒有出框
- 光線是否太暗或太亮
- 背景是否影響辨識
- gloss 順序是否跟 template 一樣
- 動作是否太快、太亂、斷掉

若需重錄：

- 在 checklist 中將 `need_rerecord` 標成 `Y`
- 在 `notes` 寫清楚原因

審片組要用到的檔案：

- [sequence_recording_manifest_300.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_recording_manifest_300.csv)
- [sequence_review_checklist_300.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_review_checklist_300.csv)

---

### 3. 標註組

負責內容：

- 依影片內容填寫 gloss sequence 標註
- 填每個 gloss 的 `start_sec` / `end_sec`
- 確認 label 名稱是否與 glossary 一致

標註方式：

- 一支影片會對應多列
- 每一列只標一個 gloss
- `order` 代表該 gloss 在影片中的順序
- `done` 填 `Y` 代表該列已完成確認

標註原則：

- 先確認 gloss 順序正確
- 再填時間
- 秒數建議填到小數三位
- 不確定的地方先寫在 `notes`

標註組要用到的檔案：

- [sequence_recording_manifest_300.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_recording_manifest_300.csv)
- [sequence_annotations_300_template.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_annotations_300_template.csv)
- [glossary_template.csv](C:\Users\User\Downloads\Temp\data\annotations\glossary_template.csv)

---

### 4. Gloss Description + 文件整理組

負責內容：

- 整理所有 gloss 的正式名稱
- 整理 alias / 同義寫法
- 撰寫每個 gloss 的動作描述
- 整理 glossary，作為後續 HST-SLR 使用

gloss description 要寫什麼：

- 這個 gloss 的意思
- 手勢大致怎麼做
- 動作方向
- 手部位置或變化
- 若有別名，也要一起記錄

例如：

- `冷氣團`：雙手由上往下帶出冷空氣壓下來的感覺
- `北部`：手勢指向上方位置表示北部
- `其他`：本專題中限定為「其他地區」的其他

Description 組要用到的檔案：

- [glossary_template.csv](C:\Users\User\Downloads\Temp\data\annotations\glossary_template.csv)
- [gloss_description_assignment.csv](C:\Users\User\Downloads\Temp\data\annotations\gloss_description_assignment.csv)

---

## 已建立好的 CSV 檔案

1. [sequence_templates_12.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_templates_12.csv)
2. [sequence_recording_manifest_300.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_recording_manifest_300.csv)
3. [sequence_review_checklist_300.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_review_checklist_300.csv)
4. [sequence_annotations_300_template.csv](C:\Users\User\Downloads\Temp\data\annotations\sequence_annotations_300_template.csv)
5. [glossary_template.csv](C:\Users\User\Downloads\Temp\data\annotations\glossary_template.csv)
6. [gloss_description_assignment.csv](C:\Users\User\Downloads\Temp\data\annotations\gloss_description_assignment.csv)

---

## 後續流程

1. 錄影組依模板錄完影片
2. 審片組檢查是否需要補錄
3. 標註組填 gloss 與時間
4. Description 組完成 glossary 與 gloss 描述
5. 再把完整資料接進後續 HST-SLR / sequence 模型訓練
