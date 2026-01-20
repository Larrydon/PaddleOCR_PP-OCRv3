# 自動分離 train / val（8:2）

import random
from pathlib import Path

# ================= 設定 =================
DATA_DIR = Path("./train_data")
SRC_LIST = DATA_DIR / "train_list.txt"
TRAIN_LIST = DATA_DIR / "train_list.txt"
VAL_LIST = DATA_DIR / "val_list.txt"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
# =======================================


def split_train_val():
    with open(SRC_LIST, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total = len(lines)
    if total < 5:
        raise ValueError("資料筆數過少，無法切分 train / val")

    print(f"目前是 {SRC_LIST} 總共:{total}筆 SimpleDataSet")

    random.seed(RANDOM_SEED)
    random.shuffle(lines)

    train_count = int(total * TRAIN_RATIO)
    train_lines = lines[:train_count]
    val_lines = lines[train_count:]

    with open(TRAIN_LIST, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(VAL_LIST, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"Total: {total}")
    print(f"Train: {len(train_lines)}")
    print(f"Val  : {len(val_lines)}")
    print("Split completed.")


if __name__ == "__main__":
    split_train_val()
