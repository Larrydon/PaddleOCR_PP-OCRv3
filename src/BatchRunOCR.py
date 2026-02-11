import os
import glob
from YOLOv8OCR import process_single_image  # 匯入剛封裝的函數
import time

# ================== 設定 ==================
INPUT_FOLDER = "./BatchRunOCRs/input_images"  # 你的圖檔根目錄
OUTPUT_FOLDER = "./BatchRunOCRs/ocr_result"  # 結果儲存目錄
IMAGE_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


def main():
    # 1. 取得所有圖檔路徑
    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    if not image_files:
        print(f"❌ 在 {INPUT_FOLDER} 找不到任何圖片。")
        return

    print(f"🚀 開始批次處理，共 {len(image_files)} 張圖...")
    start_time = time.time()

    # 2. 逐一處理
    for i, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_files)}] 正在處理: {img_name}")

        try:
            results, saved_path = process_single_image(img_path, OUTPUT_FOLDER)

            # 顯示結果
            if results:
                for r in results:
                    # ⚠️ 修正點：key 名稱要跟 YOLOv8OCR 裡面定義的一樣 (用 'plate')
                    print(f"   ✨ 辨識到: {r['plate']} (信心度: {r['score']:.2f})")
            else:
                print("   ⚠️ 未偵測到車牌")

        except Exception as e:
            import traceback

            print(f"   ❌ 處理失敗: {img_name}")
            print(traceback.format_exc())  # 打印詳細錯誤以便追蹤

    end_time = time.time()
    print("-" * 30)
    print(f"✅ 批次處理完成！")
    print(f"⏱️ 總耗時: {end_time - start_time:.2f} 秒")
    print(f"📁 結果儲存在: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
