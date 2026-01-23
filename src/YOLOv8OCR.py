# YOLOv8 + PaddleOCR 2.10.0 + 車牌透視校正

from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

# ================== 設定 ==================
TEST_IMG_PATH = "./test_car.jpg"
OUTPUT_PATH = "yolo8_ocr.jpg"
DEBUG_DIR = "ocr_debug"

YOLO_CONF_THRESH = 0.5  # YOLOv8 車牌信心值過濾 0.5
OCR_SCORE_THRESH = 0.5  # OCR 信心值過濾 0.5
PAD_RATIO = 0.3  # 車牌裁切安全擴大比例 0.3

FONT_TEXT = "./NotoSansCJKtc-Regular.otf"
REC_MODEL_PATH = "./ch_PP-OCRv3_rec_slim2_infer"

os.makedirs(DEBUG_DIR, exist_ok=True)

# ================== 初始化模型 ==================
yolo_model = YOLO("detectoryolov8s.pt")  # 車牌模型 detectoryolov8s.pt

ocr = PaddleOCR(
    # lang="ch", # 可以拿掉，因為您已經指定了自定義模型和字典
    det=False,
    use_angle_cls=False,
    rec_model_dir=REC_MODEL_PATH,
    use_gpu=False,
    rec_char_dict_path="./ppocr/utils/dict/dict_taiwan_car.txt",  # 必須指定字典
)  # 明確指定 ch_PP-OCRv3_rec_infer ch_PP-OCRv3_rec_slim2_infer


# 確認識別器已正確初始化
if ocr.text_recognizer is not None:
    print("✓ 識別預測器已正確載入指定模型:" + REC_MODEL_PATH)
else:
    print("✗ 識別預測器載入失敗:" + REC_MODEL_PATH)


# ================== 工具函數 ==================
def expand_box(x1, y1, x2, y2, pad_ratio, img_w, img_h):
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    )


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ],
        dtype="float32",
    )


def perspective_correction(img, boxes):
    if not boxes:
        return np.array(img)

    all_pts = np.concatenate([np.array(b) for b in boxes], axis=0)
    hull = cv2.convexHull(all_pts.astype(np.float32))
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = order_points(box)

    w = int(rect[1][0])
    h = int(rect[1][1])
    if w < h:
        w, h = h, w

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(np.array(img), M, (w, h))
    return warped


# YOLO 矩形 → 4點 polygon
def rect_to_poly(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


# 計算 polygon 交集比例
def intersection_ratio(ocr_box, yolo_rect):
    if len(ocr_box) < 4:
        return 0.0
    ocr_poly = np.array(ocr_box, dtype=np.float32)
    yolo_poly = rect_to_poly(*yolo_rect)

    area_ocr = cv2.contourArea(ocr_poly)
    if area_ocr <= 1:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(ocr_poly, yolo_poly)
    return inter_area / area_ocr if area_ocr > 0 else 0.0


def draw_yolo_only(draw, x1, y1, x2, y2, font):
    draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
    draw.text((x1, max(0, y1 - 25)), "YOLO only", fill="yellow", font=font)


def imread_equivalent(img_np):
    """
    將任意來源的 numpy image，強制轉成 PaddleOCR
    等價於 cv2.imread() 的輸入格式
    """
    # 1️⃣ 確保 numpy
    if not isinstance(img_np, np.ndarray):
        raise TypeError("input must be numpy array")

    # 2️⃣ dtype 強制 uint8
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    # 3️⃣ channel 修正
    if img_np.ndim == 2:
        # gray → BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.ndim == 3 and img_np.shape[2] == 3:
        pass
    else:
        raise ValueError(f"unsupported image shape: {img_np.shape}")

    # 4️⃣ 記憶體連續（⚠️ 非常關鍵）
    img_np = np.ascontiguousarray(img_np)

    # 5️⃣ 尺寸保護（rec 最低容忍）
    h, w = img_np.shape[:2]
    if h < 16 or w < 16:
        return None

    return img_np


def parse_rec_result(ocr_res):
    """
    適用於 det=False 的 PaddleOCR 輸出
    """
    if not ocr_res or not ocr_res[0]:
        return "", 0.0

    texts, scores = [], []
    for text, score in ocr_res[0]:
        texts.append(text)
        scores.append(score)

    return " ".join(texts), float(np.mean(scores))


# ================== 主流程 ==================
orig_img = Image.open(TEST_IMG_PATH).convert("RGB")
draw = ImageDraw.Draw(orig_img)
W, H = orig_img.size

try:
    font_default = ImageFont.truetype(FONT_TEXT, 24)
except:
    font_default = ImageFont.load_default()

yolo_results = yolo_model.predict(TEST_IMG_PATH)

try:
    font = ImageFont.truetype(FONT_TEXT, 20)
except:
    font = font_default

for idx, yolo_box in enumerate(yolo_results[0].boxes.xyxy):
    # 1️⃣ 原始 YOLO 框
    x1, y1, x2, y2 = map(int, yolo_box.tolist())

    # 2️⃣ 固定 expand_box
    x1e, y1e, x2e, y2e = expand_box(
        x1, y1, x2, y2, pad_ratio=PAD_RATIO, img_w=W, img_h=H
    )

    # 3️⃣ Crop 後第一次 OCR，取回 OCR 4點座標
    crop_img = orig_img.crop((x1e, y1e, x2e, y2e))
    crop_np = np.array(crop_img)
    ocr_res = ocr.ocr(crop_np)

    # ===== 判斷 OCR 是否有結果 =====
    if not ocr_res or not ocr_res[0]:
        # YOLO 偵測到，但 OCR 無結果 → 畫藍框
        draw_yolo_only(draw, x1, y1, x2, y2, font)
        print(f"❌ YOLO 偵測到，但 OCR 無結果: [{x1},{y1},{x2},{y2}]")
        # 跳到下一個 YOLO 框
        continue

    ocr_boxes = [line[0] for line in ocr_res[0]]

    # 將 OCR box 映射回原圖座標
    ocr_boxes_global = []
    for b in ocr_boxes:
        global_b = [[x + x1e, y + y1e] for x, y in b]
        ocr_boxes_global.append(global_b)

    # ---------- 保存第一次 OCR debug 圖 ----------
    crop_img.save(os.path.join(DEBUG_DIR, f"1.ocr_raw_{idx+1}.jpg"))

    # 4️⃣ 計算交集比例
    ratios = []
    for ocr_box in ocr_boxes_global:
        r = intersection_ratio(ocr_box, (x1e, y1e, x2e, y2e))
        ratios.append(r)
    mean_ratio = np.mean(ratios) if ratios else 0

    # 5️⃣ 判斷是否做透視校正
    if mean_ratio < 0.3:
        # OCR 幾乎不在 expand_box 中 → 畫藍框
        draw_yolo_only(draw, x1, y1, x2, y2, font)

        print(f"❌ OCR 與 YOLO 框重疊率太低: [{x1},{y1},{x2},{y2}]")
        continue
    elif mean_ratio < 0.6:
        final_ocr_res = ocr_res
    else:
        # ---------- 透視校正 ----------
        corrected = perspective_correction(crop_img, ocr_boxes)
        # corrected_img = Image.fromarray(corrected)
        # corrected_img.save(f"{DEBUG_DIR}/2.perspective_{idx+1}.jpg")
        # final_ocr_res = ocr.ocr(np.array(corrected_img))
        #
        # result = ocr.ocr(f"{DEBUG_DIR}/2.perspective_{idx+1}.jpg", det=False, cls=False)
        # for idx in range(len(result)):
        #     res = result[idx]
        #     for line in res:
        #         print(line)

        # === cv2.imread 等價處理 ===
        corrected_np = imread_equivalent(corrected)

        if corrected_np is None:
            draw_yolo_only(draw, x1, y1, x2, y2, font)
            print("❌ 透視校正後影像過小，略過")
            continue

        cv2.imwrite(f"{DEBUG_DIR}/2.perspective_{idx+1}.jpg", corrected_np)
        final_ocr_res = ocr.ocr(corrected_np, det=False, cls=False)

    if not final_ocr_res or not final_ocr_res[0]:
        # OCR 無結果 → 畫藍框
        draw_yolo_only(draw, x1, y1, x2, y2, font)

        print(f"❌ OCR 辨識結果為空: [{x1},{y1},{x2},{y2}]")
        continue

    """
    只在 ocr.ocr(det=True) 時成立
    每一筆回傳是：
    [box, (text, score)]
    """
    # texts, scores = [], []
    # for _, (text, score) in final_ocr_res[0]:
    #     texts.append(text)
    #     scores.append(score)

    # avg_score = np.mean(scores) if scores else 0
    # plate_text = " ".join(texts)
    plate_text, avg_score = parse_rec_result(final_ocr_res)

    # ---------- 畫回原圖 ----------
    # ===== 根據 OCR 信心畫框顏色 =====
    if avg_score < OCR_SCORE_THRESH:
        draw.rectangle([x1, y1, x2, y2], outline="orange", width=3)
        draw.text(
            (x1, max(0, y1 - 25)),
            f"OCR low ({avg_score:.2f})",
            fill="yellow",
            font=font,
        )
    else:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text(
            (x1, max(0, y1 - 25)),
            f"{plate_text} ({avg_score:.2f})",
            fill="yellow",
            font=font,
        )

    print(
        f"✅ 辨識結果: {plate_text} | 置信度: {avg_score:.4f} | 座標: [{x1},{y1},{x2},{y2}]"
    )

    print(f"✅ 已產生第一次 OCR debug 圖：1.ocr_debug/ocr_raw_*.jpg")
    print(f"✅ 已產生透視校正 debug 圖：2.ocr_debug/perspective_*.jpg")

# ================== 輸出 ==================
orig_img.save(OUTPUT_PATH)
print(f"✅ 已輸出辨識結果：{OUTPUT_PATH}")
