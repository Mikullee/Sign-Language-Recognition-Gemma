# Branch Strategy

## `main`

穩定展示版與整合版。放目前最能代表專題成果的結構、文件與主流程。

## `feature/recognition-*`

辨識端功能開發分支。包含模型、推論流程、即時 UI、評估方式等調整。

## `feature/generation-*`

生成端功能開發分支。包含 prompt、pipeline、辨識輸出轉換與展示流程。

## `dev-agent-sync`

歷史資料同步分支，保留追溯用途，不再作為正式資料來源或主要工作分支。
