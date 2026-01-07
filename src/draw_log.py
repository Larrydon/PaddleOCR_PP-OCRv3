import re
import matplotlib.pyplot as plt

# 確保這個路徑指向您的訓練日誌檔案
LOG_Dir = "./output/rec_ppocr_v3_distillation/"
LOG_FILE = LOG_Dir + "train.log"
# 正則表達式：匹配包含 loss: X.XXXXX 的行
# 這裡假設 loss 總是在 'loss:' 之後
LOSS_PATTERN = re.compile(r"global_step: (\d+).*?loss: ([\d.]+)")

global_steps = []
losses = []

try:
    with open(LOG_FILE, "r") as f:
        for line in f:
            match = LOSS_PATTERN.search(line)
            if match:
                # match.group(1) 是 global_step
                # match.group(2) 是 loss
                global_steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
except FileNotFoundError:
    print(f"錯誤：找不到日誌檔案 {LOG_FILE}")
    exit()

if not global_steps:
    print("未找到有效的 Loss 數據。")
    exit()

# 繪製圖表
plt.figure(figsize=(10, 5))
plt.plot(global_steps, losses, label="Total Loss", marker="o", linestyle="-")
plt.title("Training Loss Over Global Steps")
plt.xlabel("Global Step")
plt.ylabel("Loss Value")
plt.grid(True)
plt.legend()
plt.savefig(LOG_Dir + "training_loss_plot.png")
print("Loss 曲線圖已保存為 training_loss_plot.png")
