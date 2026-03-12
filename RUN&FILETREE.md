# 專案結構文件樹

- 專案結構文件樹（src）VSCode run Python
				|----AutoSplitTrainVal_List.py	自動將 train_data/train_list.txt內容 82分成訓練集(train_list.txt)和驗證集(val_list.txt)
				|----BatchRunOCR.py	批次測試OCR結果
				|----check_model.py	檢查抽取CTC的模型是否正確
				|----draw_log.py	將訓練的log(train.log)整理成圖檔(training_loss_plot.png)
				|----OCR_Model_View.py	模擬模型OCR，模擬PaddleOCR訓練使用的圖檔
				|----student_model_stripped.py	將蒸餾模型抽出，另存成CTC單一模組使用
				|----testPaddleOCR.py	只用PaddleOCR REC模式，檢測辨識模型
				|----YOLOv8OCR.py	主要程式，實現YOLO找車牌->PaddleOCR 辨識車牌
 
- `.gitignore`
- `CHANGELOG.md`
- `README.md`
- `RUN&FILETREE.md`