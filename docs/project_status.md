# Project Status

更新日期：2026-07-16

## 現在做到哪

目前主成果是 fixed sentence 辨識版本，已整理出 28 個固定常用句子的辨識流程，包含資料整理、MediaPipe 特徵抽取、BiGRU 訓練、offline 評估與即時測試。

## Fixed Sentence 現況

- 以 daily30 sentence 資料為主
- 主模型為 BiGRU sentence classifier
- 目前已有 webcam / mp4 測試工具
- 已加入 top3 候選顯示與 session log
- manual trigger 可用，auto-trigger 持續調整中

## Gloss 現況

- 已做 gloss 版本與相關訓練實驗
- 目前效果弱於固定句型版本
- 初步判斷原因是 gloss 較短、動作幅度與語意特徵較不穩定，對 GRU 學習較不利

## Auto-Trigger 現況

- 目標是取代手動按空白鍵開始 / 結束
- 目前已加入規則式 BLANK / SIGNING 切段
- 桌面遮擋、手部不可見、邊界過短等情境仍在調整

## 下一步

- 持續優化 auto-trigger 的開始 / 結束穩定度
- 讓 fixed sentence realtime demo 更穩定
- 規劃辨識端輸出如何接到生成端
