# 更新日誌

專案的版本更新內容會被記錄在這個檔案

更新日誌的格式將會基於 [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
==============================================================================
## [1.5.0] - 2026-02-11
### Fixed
- `棄用蒸餾訓練，改成單純 CTC訓練 image_shape=[3, 48, 192]，max_text_length 長度可以改成10碼就好`

### Updated
- `更新 YOLOv8OCR.py 改成可以提供函式呼叫)`

### Added
- `新增 BatchRunOCR.py 用來直接跑批次圖片車牌辨識，方便看模型辨識後的正確與否，直接顯示在圖片上`


## [1.4.0] - 2026-02-03
### Fixed
- `修正 YOLOv8OCR.py ocr初始化方式和流程
- `1.用YOLO找出車牌並用OCR DET 找出車牌4點`
- `2.4點座標做透視水平修正`
- `2.偵測不到的或是交集不夠的，使用多角度辨識回傳最佳結果`
- `3.OCR REC 辨識車牌`
- `4.畫回原圖`

### Fixed
- `修正 字典 dict_taiwan_car.txt`

### Added
- `新增 OCR_Model_View.py 查看模型在辨識用的圖長什麼樣子和顯示長寬比例`


## [1.3.0] - 2026-01-20
### Added
- `新增 src\AutoSplitTrainVal_List.py 用來自動按照8:2分成 .\train_data\train_list.txt 和 .\train_data\val_list.txt`
- `新增 src\configs\rec\PP-OCRv3\rec_carplate_train_gpu_2.yml 新增第二輪增強訓練設定檔`

### Updated
- `更新 src\configs\rec\PP-OCRv3\rec_carplate_train_gpu.yml 開始真實的訓練(5千張以內都還算是小型模型)耗時5個半小時(116筆真實照片+1050筆合成照片=總共1166筆)`
- `更新 src\configs\rec\PP-OCRv3\rec_carplate_train_gpu.yml src\configs\rec\PP-OCRv3\rec_carplate_train_cpu.yml 參數註解`

### Fixed
- `修復 src\YOLOv8OCR.py 解決 ocr.ocr(np.array(corrected_img)) 辨識不到，cv2.imread 等價處理就可以；因為OpenCV / PIL 在讀檔時幫你做了「格式標準化」`


## [1.2.0] - 2026-01-16
### Updated
- `src/.vscode/launch.json 新增 tools\infer_rec.py、tools\export_model.py、tools\infer\predict_rec.py`


## [1.1.0] - 2026-01-13
### Added
- `src\configs\rec\PP-OCRv3\rec_carplate_train_cpu.yml`
- `src\configs\rec\PP-OCRv3\rec_carplate_train_gpu.yml`
- `src\draw_log.py`
- `tools\train.py 訓練完成後，會得到動態圖模型 best_accuracy.pdparams`


## [1.0.0] - 2025-12-23
### Added
- `專案結構文件樹->RUN&FILETREE.md`
- `#原始碼src: Fork from [PaddleOCR-2.10.0](https://github.com/PaddlePaddle/PaddleOCR/tree/v2.10.0)`


