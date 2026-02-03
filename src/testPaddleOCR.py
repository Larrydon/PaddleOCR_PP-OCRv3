# 只測試OCR辨識功能 rec

from paddleocr import PaddleOCR


REC_MODEL_PATH = "./ch_PP-OCRv3_rec_slim2_infer"

ocr = PaddleOCR(
    # lang="ch", # 可以拿掉，因為您已經指定了自定義模型和字典
    det=False,
    rec=True,
    cls=False,
    rec_model_dir=REC_MODEL_PATH,
    rec_char_dict_path="./ppocr/utils/dict/dict_taiwan_car.txt",  # 必須指定字典
    use_gpu=False,
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
