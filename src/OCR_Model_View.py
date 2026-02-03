import cv2
import numpy as np
import os

# 注意：模型輸入是 320 寬，48 高
TARGET_H, TARGET_W = 48, 128
VIEW_IMG_FILE = "./my_synthetic_image.jpg"  # "./my_synthetic_image.jpg"


def visualize_paddle_input(
    image_path, target_h=48, target_w=128, save_name="model_view.jpg"
):
    # 1. 讀取原始裁切照
    img = cv2.imread(image_path)
    if img is None:
        print("找不到圖片，請檢查路徑")
        return

    # 2. 取得原始尺寸與比例
    ori_h, ori_w = img.shape[:2]
    ori_ratio = ori_w / ori_h  # 原始比例

    # 3. 計算模型目標比例
    target_ratio = TARGET_W / TARGET_H  # 模型看到的真實比例 (例如 128/48 = 2.67)

    # 4. 模擬 PaddleOCR 的 Resize 邏輯
    # 注意：這裡模擬的是直接強拉 (Distortion)，而非 Padding 模式
    img_resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

    # 5. 計算變形率 (Distortion Rate)
    # 如果值接近 1.0，代表比例保持得很好；偏離越多，字體變形越嚴重
    distortion_rate = ori_ratio / target_ratio

    # 6. 儲存結果
    cv2.imwrite(save_name, img_resized)

    print(f"--- 比例分析報告 ---")
    print(f"原始尺寸: {ori_w}x{ori_h} (比例: {ori_ratio:.2f})")
    print(f"模型輸入: {target_w}x{target_h} (比例: {target_ratio:.2f})")

    if abs(1 - distortion_rate) < 0.1:
        print(f"狀態: ✅ 比例基本維持 (變形率: {distortion_rate:.2f})")
    else:
        status = "拉長" if distortion_rate < 1 else "壓扁"
        print(f"狀態: ⚠️ 嚴重變形 - 字體被{status} (變形率: {distortion_rate:.2f})")

    print(f"可視化圖片已存為: {save_name}")


# 調用範例
# visualize_paddle_input("crop_0.jpg", target_h=48, target_w=128)

# 請替換成您的合成照路徑
visualize_paddle_input(VIEW_IMG_FILE, target_h=TARGET_H, target_w=TARGET_W)
