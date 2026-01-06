# 只測試OCR辨識功能 rec

from paddleocr import PaddleOCR


REC_MODEL_PATH = "./ch_PP-OCRv3_rec_slim2_infer"

ocr = PaddleOCR(
    lang="ch", rec_model_dir=REC_MODEL_PATH, device="cpu"
)  # 明確指定 ch_PP-OCRv3_rec_infer ch_PP-OCRv3_rec_slim2_infer
# print("識別模型:", ocr.rec_model_config)

# 確認識別器已正確初始化
if ocr.text_recognizer is not None:
    print("✓ 識別預測器已正確載入指定模型:" + REC_MODEL_PATH)
else:
    print("✗ 識別預測器載入失敗:" + REC_MODEL_PATH)

img_path = "./word_10.png"
result = ocr.ocr(img_path, det=False, cls=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)
