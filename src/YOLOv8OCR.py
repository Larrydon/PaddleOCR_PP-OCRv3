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
YOLO_PAD_RATIO = 0.3  # 車牌裁切安全擴大比例 0.3
ROTATE_ANGLES = [2, 10, 15, -10, -15]  # 可調旋轉補償角度
INTERSECTION_GOOD = 0.6


FONT_TEXT = "./NotoSansCJKtc-Regular.otf"
# DET_MODEL_PATH = "./ocr_models/det/ch_PP-OCRv3_det_slim2_infer"
REC_MODEL_PATH = "./ocr_models/rec/ch_PP-OCRv3_rec_slim2_infer"
os.makedirs(DEBUG_DIR, exist_ok=True)


# ================== 初始化模型 ==================
yolo_model = YOLO("detectoryolov8s.pt")  # 車牌模型 detectoryolov8s.pt


# 官方不建議的混用方式（初始化和呼叫方式不同），因此DET和REC都先初始化=True
pocr = PaddleOCR(
    # lang="ch", # 可以拿掉，因為您已經指定了自定義模型和字典
    det=True,
    rec=True,
    cls=False,
    # det_model_dir=DET_MODEL_PATH,
    rec_model_dir=REC_MODEL_PATH,
    rec_char_dict_path="./ppocr/utils/dict/dict_taiwan_car.txt",  # 必須指定字典
    use_gpu=False,
    # ✨ 關鍵點：在這裡明確指定你訓練時的形狀 [通道, 高度, 寬度]
    rec_image_shape="3, 48, 192",
)  # 明確指定 ch_PP-OCRv3_rec_infer ch_PP-OCRv3_rec_slim2_infer


# 確認識別器已正確初始化
if pocr.text_recognizer is not None:
    # 印出偵測模型 (Detection) 的實際路徑
    print(f"📍 偵測模型路徑(det): {pocr.args.det_model_dir}")

    # 印出識別模型 (Recognition) 的路徑
    print(f"📍 識別模型路徑(rec): {pocr.args.rec_model_dir}")
else:
    print("✗ 識別預測器載入失敗!!!")


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

    # --- 修正後的座標檢查與 Clip 邏輯 ---
    fixed_boxes = []
    for box in boxes:
        # 確保 box 是 numpy array 方便計算
        box_np = np.array(box, dtype="float32")
        # 限制座標不超出圖片邊界 (避免原本的 TypeError)
        box_np[:, 0] = np.clip(box_np[:, 0], 0, w - 1)
        box_np[:, 1] = np.clip(box_np[:, 1], 0, h - 1)
        fixed_boxes.append(box_np)

    # 1. 取得所有點的凸包 (Convex Hull)
    all_pts = np.concatenate([np.array(b) for b in boxes], axis=0).astype(np.float32)
    hull = cv2.convexHull(all_pts)

    # --- 關鍵修改區 ---
    # 使用 approxPolyDP 逼近多邊形，直到點數剩下 4 個
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # 如果逼近出來不是 4 個點（可能是 5, 6 個），則強制取 4 個點的最小外接矩形作為保險
    if len(approx) == 4:
        box = approx.reshape(4, 2).astype(np.float32)
    else:
        # 保險機制：如果多邊形太複雜，改用更穩定的四點逼近
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)

    # 務必進行排序，確保 tl, tr, br, bl 順序正確
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
    # 如果你發現還是會切到，建議把這兩個 padding 調大
    dynamic_pad_w = int(maxWidth * 0.1)  # 增加到 10%
    dynamic_pad_h = int(maxHeight * 0.2)
    new_w = maxWidth + 2 * dynamic_pad_w
    new_h = maxHeight + 2 * dynamic_pad_h

    # 4. 設定目標座標 (Dst)
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


# YOLO 矩形->強健邊緣檢測 → 4點 polygon
def find_plate_quad_robust(crop_img, idx=0):  # 記得加上 idx 參數
    # 1. 轉灰階並強化邊緣
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # 使用中值濾波除去車牌上的螺絲或小雜訊，保留邊界線
    # blurred = cv2.medianBlur(gray, 9)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Canny 邊緣檢測
    edged = cv2.Canny(blurred, 120, 255)

    # 3. 形態學處理 (核心修正：用閉合運算接斷線並過濾雜訊)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 4. 找最大輪廓
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    # 找面積最大的輪廓
    c = max(cnts, key=cv2.contourArea)

    # 5. 取得凸包並逼近
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)

    # 稍微提高容忍度 (0.02 -> 0.03)，有助於減少多餘的點
    approx = cv2.approxPolyDP(hull, 0.03 * peri, True)

    # --- 修正：在判斷前先存下原始偵測點 (Raw Points) ---
    raw_debug = crop_img.copy()
    raw_pts = approx.reshape(-1, 2).astype(np.int32)

    # 畫出所有逼近點與連線
    cv2.polylines(raw_debug, [raw_pts], True, (255, 255, 255), 1)  # 白色細線

    for pt in raw_pts:
        cv2.circle(raw_debug, tuple(pt), 4, (255, 0, 0), -1)  # 藍色圓點

    cv2.putText(
        raw_debug,
        f"Approx count: {len(approx)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )

    cv2.imwrite(f"{DEBUG_DIR}/2_approx_raw_{idx}.jpg", raw_debug)
    debug_img = crop_img.copy()

    if len(approx) == 4:
        # --- 成功：畫綠色框 ---
        res_box = approx.reshape(4, 2).astype(np.int32)
        cv2.polylines(debug_img, [res_box], True, (0, 255, 0), 2)

        for pt in res_box:
            cv2.circle(debug_img, tuple(pt), 5, (0, 0, 255), -1)  # 紅色端點

        cv2.imwrite(f"{DEBUG_DIR}/2_find4point_{idx}.jpg", debug_img)
        return approx.reshape(4, 2).astype(np.float32)
    else:
        # --- 失敗處理：畫黃色保險框 ---
        # 關鍵修正：使用 hull (凸包) 算矩形，才不會被掉在中間的藍點拉扯
        rect = cv2.minAreaRect(hull)
        box_points = cv2.boxPoints(rect).astype(np.int32)

        # 畫出黃色框
        cv2.polylines(debug_img, [box_points], True, (0, 255, 255), 2)

        # 把失敗的藍點也補上去對照
        for pt in raw_pts:
            cv2.circle(debug_img, tuple(pt), 3, (255, 0, 0), -1)

        cv2.putText(
            debug_img,
            f"Failed: {len(approx)} pts",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        cv2.imwrite(f"{DEBUG_DIR}/2_fallback_rect_{idx}.jpg", debug_img)
        return box_points.astype(np.float32)


def save_debug_plate_image(crop_img, plate_box, idx, debug_dir):
    """
    在截圖上畫出偵測到的四個角點與邊框，並儲存 debug 影像。
    """

    # 1. 為了不破壞原始截圖，我們複製一份來畫圖
    # 如果 crop_img 是灰階，轉成 BGR 方便畫彩色線條
    if len(crop_img.shape) == 2:
        debug_canvas = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
    else:
        debug_canvas = crop_img.copy()

    # 2. 檢查 plate_box 是否有值
    if plate_box is not None:
        # 確保座標是整數，OpenCV 繪圖函數通常要求整數型態
        # plate_box 預期形狀為 (4, 2)
        pts = plate_box.astype(np.int32)

        # 3. 畫出四邊形的邊框 (綠色)
        # pts.reshape((-1, 1, 2)) 是 cv2.polylines 的標準格式

        cv2.polylines(
            debug_canvas,
            [pts.reshape((-1, 1, 2))],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        # 4. 畫出四個角點 (紅色圓點)
        # 順便標註 0, 1, 2, 3 序號，方便你檢查頂點順序（對透視變換很重要）
        for i, pt in enumerate(pts):
            cv2.circle(debug_canvas, (pt[0], pt[1]), 5, (0, 0, 255), -1)
            cv2.putText(
                debug_canvas,
                str(i),
                (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    # 5. 儲存結果
    output_path = f"{debug_dir}/2_find4point_{idx}.jpg"
    cv2.imwrite(output_path, debug_canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print(f"Debug image saved to: {output_path}")


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
    # 1. 檢查結構是否為空
    if not ocr_res or not isinstance(ocr_res, list) or len(ocr_res) == 0:
        return "", 0.0

    # PaddleOCR det=False 時，結果通常在 ocr_res[0]
    res_list = ocr_res[0]

    if not res_list:
        return "", 0.0

    texts = []
    scores = []

    for item in res_list:
        try:
            # 正常情況 item 應該是 ('ABC-123', 0.98)
            if isinstance(item, (list, tuple)) and len(item) == 2:
                t, s = item
                texts.append(str(t))
                scores.append(float(s))
            else:
                # 這裡就是導致你失敗的地方：如果 item 不是兩個值，就會跳到這
                print(f"⚠️ 偵測到異常 OCR 結構: {item}")
                continue
        except Exception as e:
            print(f"⚠️ 解析單項 OCR 結果失敗: {e}, 數據內容: {item}")
            continue

    if not texts:
        return "", 0.0

    return " ".join(texts), float(np.mean(scores))


def rotate_image(img, angle):
    if abs(angle) < 1e-3:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def ocr_det_valid(ocr_res, min_boxes=2):
    if not ocr_res or not ocr_res[0]:
        return False
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


def draw_text_with_outline(
    draw, position, text, font, text_color="yellow", outline_color="black"
):
    x, y = position
    # 畫四個角落的黑字
    for offset in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((x + offset[0], y + offset[1]), text, fill=outline_color, font=font)
    # 畫中間的亮色字
    draw.text((x, y), text, fill=text_color, font=font)


# ================== 主流程 ==================
# ================== 封裝後的辨識函式 ==================
def process_single_image(img_path, output_dir="results"):
    """
    輸入：圖片路徑, 輸出資料夾
    輸出：(結果列表, 儲存路徑)
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(img_path)
    final_output_path = os.path.join(output_dir, f"res_{filename}")

    # ⚠️ 修正點：必須使用傳入的 img_path，而不是 TEST_IMG_PATH
    try:
        orig_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"無法開啟圖片 {img_path}: {e}")
        return [], None

    draw = ImageDraw.Draw(orig_img)
    W, H = orig_img.size

    # 準備存儲這張圖的所有結果
    summary_results = []

    try:
        font_default = ImageFont.truetype(FONT_TEXT, 20)
    except:
        font_default = ImageFont.load_default()

    yolo_results = yolo_model.predict(img_path, verbose=False)
    for idx, yolo_box in enumerate(yolo_results[0].boxes.xyxy):
        # 原始 YOLO 框
        x1, y1, x2, y2 = map(int, yolo_box.tolist())

        # 固定 expand_box
        """
        debug 4點座標 => ocr_boxes_local
        """

        x1e, y1e, x2e, y2e = expand_box(x1, y1, x2, y2, YOLO_PAD_RATIO, W, H)
        crop_img = orig_img.crop((x1e, y1e, x2e, y2e))
        crop_np = np.array(crop_img)
        crop_img.save(f"{DEBUG_DIR}/1_crop_{idx}.jpg", quality=100)

        # 使用 PaddleOCR DET來偵測車牌4點，用這4點做 透視校正
        ocr_det_res = pocr.ocr(crop_np, det=True, cls=False)
        draw_det_debug(crop_img, ocr_det_res, f"{DEBUG_DIR}/2_det_{idx}.jpg")

        ocr_det_ok = ocr_det_valid(ocr_det_res)
        if ocr_det_ok:
            ocr_boxes_local = [line[0] for line in ocr_det_res[0]]
            ocr_boxes_global = [
                [[x + x1e, y + y1e] for x, y in box] for box in ocr_boxes_local
            ]

            mean_ratio = mean_intersection_ratio(ocr_boxes_global, (x1e, y1e, x2e, y2e))
        else:
            mean_ratio = 0.0

        """
        debug 4點座標 => ocr_boxes_local
        """
        ocr_det_ok = True
        # 你的原始資料
        ocr_boxes_local = [[17.0, 23.0], [94.0, 15.0], [108.0, 57.0], [32.0, 61.0]]
        # 組成函式需要的格式：[box1, box2, ...]
        # 即使只有一個 box，也要包成 list
        ocr_boxes_local = [ocr_boxes_local]

        # ===== OCR決策 =====
        """
        方法一:YOLO找出車牌
        1.PaddleOCR(DET) 和 YOLO算交集+透視校正+PaddleOCR(REC)
        2.PaddleOCR(DET)找不到或是沒交集，使用多角度做PaddleOCR(REC)找出最佳值
        """
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
            # 多角度旋轉補償 + OCR feedback
            best_text = ""
            best_score = 0
            best_img = crop_np  # 預設使用原始裁切圖
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

            ocr_input = imread_equivalent(rotated)
            avg_score = best_score
            plate_text = best_text
        # #
        # #
        # """
        # 方法二:YOLO找出車牌
        # CPU算出車牌4個點(OpenCV)
        # """
        # plate_box = find_plate_quad_robust(crop_np, idx)
        # save_debug_plate_image(crop_np, plate_box, idx, DEBUG_DIR)
        # # 執行透視校正
        # if plate_box is not None:
        #     try:
        #         # 這裡執行校正並存給 corrected 變數
        #         corrected = perspective_correction_skew_adaptive(crop_np, [plate_box])
        #     except Exception as e:
        #         print(f"校正發生錯誤：{e}")
        #         corrected = crop_np
        # else:
        #     corrected = crop_np
        #
        # # 儲存校正後的結果 (3_perspective_idx.jpg)
        #
        # cv2.imwrite(
        #     f"{DEBUG_DIR}/3_perspective_{idx}.jpg",
        #     corrected,
        #     [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        # )
        #
        # ocr_input = imread_equivalent(corrected)
        #
        # # ===== OCR rec =====
        # final_ocr_res = pocr.ocr(ocr_input, det=False, cls=False)
        # plate_text, avg_score = parse_rec_result(final_ocr_res)
        #
        # cv2.imwrite(
        #     f"{DEBUG_DIR}/4_rec_{idx}.jpg",
        #     ocr_input,
        #     [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        # )

        # 存入摘要結果
        summary_results.append({"plate": plate_text, "score": avg_score})
        #
        #
        # ---------- 畫回原圖 ----------
        # ===== 根據 OCR 信心畫框顏色 =====
        if avg_score < OCR_SCORE_THRESH:
            draw.rectangle([x1, y1, x2, y2], outline="orange", width=3)

            # draw.text(
            #     (x1, max(0, y1 - 25)),
            #     f"OCR low ({avg_score:.2f})",
            #     fill="yellow",
            #     font=font_default,
            # )
        else:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # draw.text(
            #     (x1, max(0, y1 - 25)),
            #     f"{plate_text} ({avg_score:.2f})",
            #     fill="yellow",
            #     font=font_default,
            # )

        draw_text_with_outline(
            draw,
            (x1, max(0, y1 - 25)),
            f"{plate_text} ({avg_score:.2f})",
            font_default,
        )

        print(
            f"✅ 辨識結果: {plate_text} | 置信度: {avg_score:.4f} | 座標: [{x1},{y1},{x2},{y2}]"
        )

    # ================== 輸出 ==================
    orig_img.save(final_output_path, quality=100)
    print(f"✅ 已輸出辨識結果：{final_output_path}")
    # ⚠️ 關鍵：回傳 格式化後的清單 與 檔案路徑
    return summary_results, final_output_path


# 這裡保留一個測試區塊，當你直接執行 YOLOv8OCR.py 時才會跑
if __name__ == "__main__":
    test_res = process_single_image("./test_car.jpg")
    print(f"測試結果: {test_res}")
