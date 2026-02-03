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
ROTATE_ANGLES = [2, 10, 15, -10, -15]  # 可調旋轉補償角度
INTERSECTION_GOOD = 0.6


FONT_TEXT = "./NotoSansCJKtc-Regular.otf"
REC_MODEL_PATH = "./ch_PP-OCRv3_rec_slim2_infer"

os.makedirs(DEBUG_DIR, exist_ok=True)

# ================== 初始化模型 ==================
yolo_model = YOLO("detectoryolov8s.pt")  # 車牌模型 detectoryolov8s.pt

# 官方不建議的混用方式（初始化和呼叫方式不同），因此DET和REC都先初始化=True
pocr = PaddleOCR(
    # lang="ch", # 可以拿掉，因為您已經指定了自定義模型和字典
    det=True,
    rec=True,
    cls=False,
    rec_model_dir=REC_MODEL_PATH,
    rec_char_dict_path="./ppocr/utils/dict/dict_taiwan_car.txt",  # 必須指定字典
    use_gpu=False,
)  # 明確指定 ch_PP-OCRv3_rec_infer ch_PP-OCRv3_rec_slim2_infer


# 確認識別器已正確初始化
if pocr.text_recognizer is not None:
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


# 透視水平，拉伸比例
def perspective_correction(img, boxes, padding_ratio=0.05):
    if not boxes:
        return np.array(img)

    all_pts = np.concatenate([np.array(b) for b in boxes], axis=0)
    hull = cv2.convexHull(all_pts.astype(np.float32))
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = order_points(box)

    # --- 新增：頂點外擴邏輯 (Padding) ---
    # 計算矩形的寬高，用來決定外擴的絕對像素值
    w_rect = rect[1][0]
    h_rect = rect[1][1]
    if w_rect < h_rect:
        w_rect, h_rect = h_rect, w_rect

    offset_w = w_rect * padding_ratio  # 左右各往外推 5%
    offset_h = h_rect * padding_ratio  # 上下各往外推 5%

    # 重新定義目標尺寸 (Dst) 也要加上這些 Offset
    new_w = int(w_rect + 2 * offset_w)
    new_h = int(h_rect + 2 * offset_h)

    # 這裡的關鍵是：dst 的座標要從 (offset_w, offset_h) 開始
    # 這樣變換後的圖，四周就會留出我們預設的空白，不會切到字
    dst = np.array(
        [[0, 0], [new_w - 1, 0], [new_w - 1, new_h - 1], [0, new_h - 1]],
        dtype="float32",
    )

    # 我們保持原始 box 不變，但讓 dst 變大，並調整對齊位置
    # 或是更簡單的方法：直接修改 box 讓它往外擴
    M = cv2.getPerspectiveTransform(
        box,
        np.array(
            [
                [offset_w, offset_h],
                [new_w - offset_w - 1, offset_h],
                [new_w - offset_w - 1, new_h - offset_h - 1],
                [offset_w, new_h - offset_h - 1],
            ],
            dtype="float32",
        ),
    )

    warped = cv2.warpPerspective(np.array(img), M, (new_w, new_h))
    return warped


# 透視校正傾斜自動適應
def perspective_correction_skew_adaptive(img, boxes):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not boxes:
        return img

    h, w = img.shape[:2]
    for box in boxes:
        for x, y in box:
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError("❌ perspective 收到非 local 座標")

    # 1. 取得最小外接矩形與頂點
    all_pts = np.concatenate([np.array(b) for b in boxes], axis=0)
    hull = cv2.convexHull(all_pts.astype(np.float32))
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = order_points(box)

    # 2. 計算原始車牌的「理想」寬高
    (tl, tr, br, bl) = box
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    maxWidth = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    maxHeight = int(max(height_left, height_right))

    # 3. 🎯 核心修正：計算「歪斜補償量」
    # 如果車牌很斜，maxWidth 和 maxHeight 已經不足以容納轉正後的邊角
    # 我們根據寬高比例，額外增加 20%~30% 的動態緩衝空間
    dynamic_pad_w = int(maxWidth * 0.05)  # 增加左右緩衝
    dynamic_pad_h = int(maxHeight * 0.2)  # 增加上下緩衝（針對高度歪斜）

    new_w = maxWidth + 2 * dynamic_pad_w
    new_h = maxHeight + 2 * dynamic_pad_h

    # 4. 設定目標座標 (Dst)
    # 將原始的 box 映射到這個大畫布的正中央
    dst = np.array(
        [
            [dynamic_pad_w, dynamic_pad_h],
            [new_w - dynamic_pad_w - 1, dynamic_pad_h],
            [new_w - dynamic_pad_w - 1, new_h - dynamic_pad_h - 1],
            [dynamic_pad_w, new_h - dynamic_pad_h - 1],
        ],
        dtype="float32",
    )

    # 5. 執行變換
    M = cv2.getPerspectiveTransform(box, dst)

    # 使用 BORDER_REPLICATE 填充那些「斜出去」後留下的空隙
    warped = cv2.warpPerspective(
        img, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE
    )

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


# === cv2.imread 等價處理 ===
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
    if h < 1 or w < 1:
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


# yolo旋轉校正函式，非透視
def rotate_by_yolo_crop(img_bgr):
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    coords = np.column_stack(np.where(edges > 0))
    if coords is None or len(coords) < 50:
        return img_bgr

    try:
        rect = cv2.minAreaRect(coords)
        (w, h) = rect[1]
        angle = rect[2]

        # 將 angle 正規化成「相對水平」
        if w < h:
            angle = angle + 90
    except Exception:
        return img_bgr

    # 🔒 防呆 1：寬高過小
    if w < 1 or h < 1:
        return img_bgr

    # 🔒 防呆 2：車牌必須是「橫向」
    if w < h:
        # 長邊是垂直的 → 直接放棄旋轉
        return img_bgr

    if w / h < 2.0:
        return img_bgr

    # 🔧 OpenCV angle 正規化
    if angle < -45:
        angle = 90 + angle

    # 🔒 防呆 3：角度太小不轉
    if abs(angle) < 2:
        return img_bgr

    h_img, w_img = img_bgr.shape[:2]
    center = (w_img // 2, h_img // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img_bgr,
        M,
        (w_img, h_img),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated


def rotate_image(img, angle):
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def draw_box(draw, x1, y1, x2, y2, text, score, font):
    color = "red" if score >= OCR_SCORE_THRESH else "blue"
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1, max(0, y1 - 25)), f"{text} ({score:.2f})", fill="yellow", font=font)


def ocr_det_valid(ocr_res, min_boxes=2):
    if not ocr_res or not ocr_res[0]:
        return False
    # return len(ocr_res[0]) >= min_boxes
    else:
        return True


def mean_intersection_ratio(ocr_boxes_global, yolo_rect):
    ratios = [intersection_ratio(box, yolo_rect) for box in ocr_boxes_global]
    return float(np.mean(ratios)) if ratios else 0.0


def draw_det_debug(pil_img, ocr_det_res, save_path):
    """
    將 OCR(det) 的 polygon 畫在 crop 圖上，輸出 debug 圖

    pil_img       : PIL.Image (crop_img)
    ocr_det_res   : PaddleOCR ocr(det=True) 回傳結果
    save_path    : debug 圖路徑
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    if not ocr_det_res or not ocr_det_res[0]:
        # 沒有任何 det 結果，直接存原圖
        draw.text((5, 5), "NO_OCR_DET", fill="red")
        img.save(save_path, quality=100)
        return

    for line in ocr_det_res[0]:
        box = line[0]  # 4 點 polygon
        # 將 polygon 畫成閉合線
        draw.line(box + [box[0]], fill="lime", width=2)

    img.save(save_path, quality=100)


# ================== 主流程 ==================
orig_img = Image.open(TEST_IMG_PATH).convert("RGB")
draw = ImageDraw.Draw(orig_img)
W, H = orig_img.size

try:
    font_default = ImageFont.truetype(FONT_TEXT, 20)
except:
    font_default = ImageFont.load_default()

yolo_results = yolo_model.predict(TEST_IMG_PATH)


for idx, yolo_box in enumerate(yolo_results[0].boxes.xyxy):
    # 原始 YOLO 框
    x1, y1, x2, y2 = map(int, yolo_box.tolist())

    # 固定 expand_box
    x1e, y1e, x2e, y2e = expand_box(x1, y1, x2, y2, PAD_RATIO, W, H)

    crop_img = orig_img.crop((x1e, y1e, x2e, y2e))
    crop_np = np.array(crop_img)
    crop_img.save(f"{DEBUG_DIR}/1_crop_{idx}.jpg", quality=100)

    # 使用 PaddleOCR DET來偵測車牌4點，用這4點做 透視校正
    ocr_det_res = pocr.ocr(crop_np, det=True, cls=False)
    draw_det_debug(crop_img, ocr_det_res, f"{DEBUG_DIR}/2_det_{idx}.jpg")

    # ocr_boxes_local = (
    #     [line[0] for line in ocr_det_res[0]] if ocr_det_res and ocr_det_res[0] else []
    # )

    # mean_ratio = mean_intersection_ratio(
    #     [[[x + x1e, y + y1e] for x, y in box] for box in ocr_boxes_local],
    #     (x1e, y1e, x2e, y2e),
    # )

    ocr_det_ok = ocr_det_valid(ocr_det_res)
    if ocr_det_ok:
        ocr_boxes_local = [line[0] for line in ocr_det_res[0]]
        ocr_boxes_global = [
            [[x + x1e, y + y1e] for x, y in box] for box in ocr_boxes_local
        ]
        mean_ratio = mean_intersection_ratio(ocr_boxes_global, (x1e, y1e, x2e, y2e))
    else:
        mean_ratio = 0.0

    # ===== 決策 =====
    if ocr_det_ok and mean_ratio >= INTERSECTION_GOOD:
        # 透視校正
        corrected = perspective_correction_skew_adaptive(
            np.array(crop_img), ocr_boxes_local
        )
        cv2.imwrite(
            f"{DEBUG_DIR}/3_perspective_{idx}.jpg",
            corrected,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )
        ocr_input = imread_equivalent(corrected)

        # ===== OCR rec =====
        final_ocr_res = pocr.ocr(ocr_input, det=False, cls=False)
        plate_text, avg_score = parse_rec_result(final_ocr_res)

        cv2.imwrite(
            f"{DEBUG_DIR}/4_rec_{idx}.jpg",
            ocr_input,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )
    else:
        # rotated = rotate_by_yolo_crop(crop_np)
        # 多角度旋轉補償 + OCR feedback
        best_text = ""
        best_score = 0
        for angle in ROTATE_ANGLES:
            rotated = rotate_image(crop_np, angle)
            cv2.imwrite(
                f"{DEBUG_DIR}/3_fallback_({angle})_{idx}.jpg",
                rotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
            ocr_input = imread_equivalent(rotated)
            final_ocr_res = pocr.ocr(ocr_input, det=False, cls=False)
            text, score = parse_rec_result(final_ocr_res)
            cv2.imwrite(
                f"{DEBUG_DIR}/4_rec_rotate_angle({angle})_{idx}.jpg",
                ocr_input,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
            print(f"Run rotate_angle[{idx+1}]({angle}) {text} | score={score:.2f}")
            if score > best_score:
                best_score = score
                best_text = text
                best_img = rotated
        # cv2.imwrite(f"{DEBUG_DIR}/3_rotate_{idx}.jpg", rotated)
        ocr_input = imread_equivalent(rotated)
        avg_score = best_score
        plate_text = best_text

    # ---------- 畫回原圖 ----------
    # ===== 根據 OCR 信心畫框顏色 =====
    if avg_score < OCR_SCORE_THRESH:
        draw.rectangle([x1, y1, x2, y2], outline="orange", width=3)
        draw.text(
            (x1, max(0, y1 - 25)),
            f"OCR low ({avg_score:.2f})",
            fill="yellow",
            font=font_default,
        )
    else:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text(
            (x1, max(0, y1 - 25)),
            f"{plate_text} ({avg_score:.2f})",
            fill="yellow",
            font=font_default,
        )

    print(
        f"✅ 辨識結果: {plate_text} | 置信度: {avg_score:.4f} | 座標: [{x1},{y1},{x2},{y2}]"
    )

# ================== 輸出 ==================
orig_img.save(OUTPUT_PATH, quality=100)
print(f"✅ 已輸出辨識結果：{OUTPUT_PATH}")
