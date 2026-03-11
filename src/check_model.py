import paddle
import os


# 1. 先定義路徑變數，方便後續重複使用與印出
model_path = "./pretrained/taiwan_ctc.pdparams"

# 2. 檢查路徑是否存在（除錯好習慣）
if os.path.exists(model_path):
    print(f"📂 正在從以下路徑載入模型權重: {os.path.abspath(model_path)}")
    
    # 3. 執行載入
    params = paddle.load(model_path)
    
    print("--- 權重結構檢查 ---")
    # 4. 遍歷並檢查結構
    for key in params.keys():
        print(f"Key: {key}")
        if "backbone" in key:
            print("💡 已找到 Backbone 節點，停止後續列印。")    # 只看前幾個確認結構
            break
else:
    print(f"❌ 錯誤：找不到檔案 {model_path}，請檢查路徑是否正確。")