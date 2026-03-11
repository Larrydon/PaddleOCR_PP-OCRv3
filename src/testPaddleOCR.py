# 只測試OCR辨識功能 rec

from paddleocr import PaddleOCR


REC_MODEL_PATH = "./ocr_models/rec/ch_PP-OCRv3_rec_slim2_infer"

ocr = PaddleOCR(
    # lang="ch", # 可以拿掉，因為您已經指定了自定義模型和字典
    det=False,
    rec=True,
    cls=False,
    rec_model_dir=REC_MODEL_PATH,
    rec_char_dict_path="./ppocr/utils/dict/dict_taiwan_car.txt",  # 必須指定字典
    use_gpu=False,
)  # 明確指定 ch_PP-OCRv3_rec_infer ch_PP-OCRv3_rec_slim2_infer

# 確認識別器已正確初始化
if ocr.text_recognizer is not None:
    # 印出偵測模型 (Detection) 的實際路徑
    print(f"📍 偵測模型路徑(det): {ocr.args.det_model_dir}")
    # 印出識別模型 (Recognition) 的路徑
    print(f"📍 識別模型路徑(rec): {ocr.args.rec_model_dir}")
else:
    print("✗ 識別預測器載入失敗!!!")

# img_path = "./word_10.png"
img_path = "./model_padding_view.jpg"
result = ocr.ocr(img_path, det=False, cls=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)
