# Knee42「很貴」顯示文字 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓目前 Web 版本在辨識到 `K42_26` 時一致顯示「很貴」，同時維持 class index 與既有模型相容性。

**Architecture:** 保留 `K42_26` 作為模型與資料管線的固定類別 ID，僅更新獨立的中文顯示 metadata、對應文件與 bundle digest。回歸測試同時鎖定 `K42_26 → 25` 與新顯示文字，避免日後誤把文案修改成類別重排。

**Tech Stack:** Python 3.12、unittest/pytest、JSON、CSV、SHA-256、Git。

---

### Task 1: 鎖定顯示文字與類別索引相容性

**Files:**
- Modify: `tests/test_release_safety.py`

- [ ] **Step 1: 寫入失敗的回歸測試**

在 `ReleaseSafetyTests` 加入：

```python
def test_k42_26_uses_the_revised_display_text_without_changing_its_class_index(self):
    bundle = ROOT / "artifacts" / "realtime" / "best_current"
    label_map = json.loads((bundle / "label_map_knee42.json").read_text(encoding="utf-8"))
    display_map = json.loads((bundle / "display_text_map.json").read_text(encoding="utf-8"))

    self.assertEqual(label_map["label_to_idx"]["K42_26"], 25)
    self.assertEqual(label_map["idx_to_label"][25], "K42_26")
    self.assertEqual(display_map["K42_26"], "很貴")
```

- [ ] **Step 2: 確認測試因舊顯示文字而失敗**

Run:

```powershell
python -m pytest tests/test_release_safety.py::ReleaseSafetyTests::test_k42_26_uses_the_revised_display_text_without_changing_its_class_index -q
```

Expected: FAIL，實際值為「太貴了」，而兩個 index assertion 通過。

### Task 2: 更新所有顯示來源與完整性 digest

**Files:**
- Modify: `artifacts/realtime/best_current/display_text_map.json`
- Modify: `artifacts/realtime/best_current/integrity_manifest.sha256`
- Modify: `scripts/validate_knee42_data.py`
- Modify: `docs/evaluation/live_check_42.csv`
- Modify: `README.md`

- [ ] **Step 1: 將所有受版本控制的舊顯示文字改為新文字**

只將 `K42_26` 對應的「太貴了」替換成「很貴」；不得修改 `label_map_knee42.json`、`best_model.pt` 或 checkpoint。

- [ ] **Step 2: 計算 canonical display map digest**

Run:

```powershell
python -c "import hashlib,pathlib; p=pathlib.Path(r'artifacts/realtime/best_current/display_text_map.json'); print(hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest())"
```

Expected: 輸出新的 64 字元 SHA-256。

- [ ] **Step 3: 同步 digest 記錄**

將新 digest 寫入 `integrity_manifest.sha256` 的 `display_text_map.json` 列及 README 模型 bundle 表格；其他 digest 保持不變。

- [ ] **Step 4: 確認回歸測試轉綠**

Run:

```powershell
python -m pytest tests/test_release_safety.py::ReleaseSafetyTests::test_k42_26_uses_the_revised_display_text_without_changing_its_class_index -q
```

Expected: `1 passed`。

### Task 3: 驗證目前 Web 版本並提交

**Files:**
- Modify: `tests/test_release_safety.py`
- Modify: `artifacts/realtime/best_current/display_text_map.json`
- Modify: `artifacts/realtime/best_current/integrity_manifest.sha256`
- Modify: `scripts/validate_knee42_data.py`
- Modify: `docs/evaluation/live_check_42.csv`
- Modify: `README.md`

- [ ] **Step 1: 確認所有實際來源一致且類別檔未變**

Run:

```powershell
git grep -n -I -e '太貴了' -e '很貴' -- README.md artifacts/realtime/best_current/display_text_map.json docs/evaluation/live_check_42.csv scripts/validate_knee42_data.py
git diff --exit-code ed7088e -- artifacts/realtime/best_current/label_map_knee42.json artifacts/realtime/best_current/best_model.pt
```

Expected: 指定的實際來源中舊文字零命中，新文字出現在 runtime display map、validator metadata、評估文件與 README；label map 與模型權重無差異。

- [ ] **Step 2: 執行相關測試**

Run:

```powershell
python -m pytest tests/test_knee42_transformer.py tests/test_release_safety.py tests/test_webservice.py -q
```

Expected: 顯示映射、bundle integrity 與 Web 測試通過；若 release safety 仍失敗，只能是基線已確認的 Mac 計畫文件絕對路徑。

- [ ] **Step 3: 執行完整測試與 diff 檢查**

Run:

```powershell
python -m pytest tests -q
git diff --check
git status --short
```

Expected: 本次相關測試全數通過；完整套件至多保留同一個既有基線失敗，且沒有新增失敗；diff 無 whitespace error。

- [ ] **Step 4: 提交可 cherry-pick 的單一實作 commit**

```powershell
git add -- README.md artifacts/realtime/best_current/display_text_map.json artifacts/realtime/best_current/integrity_manifest.sha256 docs/evaluation/live_check_42.csv scripts/validate_knee42_data.py tests/test_release_safety.py
git commit -m "fix: revise Knee42 expensive display text"
```

- [ ] **Step 5: 提供 Web 分支整合命令**

在原始 `fix/web-auto-trigger-live-tuning` 工作目錄完成目前未提交的 Mac 打包工具後，執行：

```powershell
$displayCommit = git rev-parse fix/knee42-display-text-expensive
git cherry-pick $displayCommit
```

再由 Web 分支的 Mac package builder 建立交付 ZIP，確保組員上傳的 Web 套件包含更新後的 bundle manifest。
