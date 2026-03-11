import cv2
import numpy as np
import os

# 注意：模型輸入是 320 寬，48 高
TARGET_H, TARGET_W = 48, 320
VIEW_IMG_FILE = "./my_synthetic_image.jpg"  # "./my_synthetic_image.jpg"
SAVE_NAME = "model_padding_view.jpg"


def get_real_transformation_report(image_path, target_h, target_w):
    img = cv2.imread(image_path)
    if img is None:
        print("找不到檔案")
        return

    h, w = img.shape[:2]

    """
    # 1. 執行 Padding Resize(Padding:false) 邏輯
    """
    # scale = target_h / h
    # new_w = int(w * scale)
    # new_w = min(new_w, target_w)

    # resized_content = cv2.resize(
    #     img, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4
    # )  # INTER_CUBIC

    # # 建立畫布並貼上內容
    # canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    # canvas[:, :new_w] = resized_content

    """
    # 1. 執行 Padding Padding:true 邏輯
    """
    # 算出等比例縮放後的寬度
    new_w = int(target_h * (w / h))

    # 進行縮放 (高度固定 48)
    if new_w > target_w:
        # 如果太寬，強行縮到 target_w
        canvas = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # 如果寬度不足 target_w，等比例縮放後，右邊補黑邊 (Padding)
        temp_img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        canvas = cv2.copyMakeBorder(
            temp_img, 0, 0, 0, target_w - new_w, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    # 3. 計算真實變形率
    ori_ratio = w / h

    # 縮放後內容比例 (在畫布上實際佔用的比例)
    # 確保計算比例時，使用的是「實際在畫布上」的寬度
    actual_w_on_canvas = min(new_w, target_w)
    current_content_ratio = actual_w_on_canvas / target_h

    # 真實變形率 = 縮放後比例 / 原始比例
    # 1.00 代表完全沒變形
    real_distortion = current_content_ratio / ori_ratio

    # 5. 儲存圖片
    cv2.imwrite(SAVE_NAME, canvas)

    # 終端機輸出報告
    print(f"--- 最終影像真實報告 ---")
    print(f"原始尺寸: {w}x{h} (比例: {ori_ratio:.2f})")
    print(f"模型畫布: {target_w}x{target_h} (比例: {current_content_ratio:.2f})")
    print(f"送入模型的真實變形率: {real_distortion:.2f}")
    print(f"✅ 偵錯圖片已存為: {SAVE_NAME}")

    if 0.98 <= real_distortion <= 1.02:
        print("狀態: ✅ 1:1 等比例送入，無變形。")
    else:
        print(f"狀態: ⚠️ 存在變形 (誤差: {abs(1-real_distortion)*100:.1f}%)")


# 請替換成您的合成照路徑
get_real_transformation_report(VIEW_IMG_FILE, TARGET_H, TARGET_W)
