import paddle
import os

# 1. 設定路徑
src_path = "./pretrained/best_accuracy.pdparams"
save_path = "./pretrained/taiwan_ctc.pdparams"

# 2. 載入權重
state_dict = paddle.load(src_path)
new_state_dict = {}

# 3. 自動對齊：只抓取 Backbone 和我們需要的 Neck/Head
for key, value in state_dict.items():
    # 關鍵：只處理 Student 權重，過濾掉佔空間的 Teacher
    if "Student." in key:
        new_key = key.replace("Student.", "")

        # 進行 Neck 與 Head 的重新映射
        # 關鍵：將蒸餾模型的 ctc_encoder 映射到我們單體模型的 Neck
        if "head.ctc_encoder" in new_key:
            new_key = new_key.replace("head.ctc_encoder", "neck")
        # 關鍵：將蒸餾模型的 ctc_head 映射到我們單體模型的 Head
        elif "head.ctc_head" in new_key:
            new_key = new_key.replace("head.ctc_head", "head")

        new_state_dict[new_key] = value
        print(f"Mapped: {key} -> {new_key}")

# 4. 儲存新權重
paddle.save(new_state_dict, save_path)
print("\n✅ 轉換完成！請使用 ./pretrained/taiwan_ctc 進行訓練")
