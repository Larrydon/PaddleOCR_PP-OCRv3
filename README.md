# Fork from [PaddleOCR-2.10.0](https://github.com/PaddlePaddle/PaddleOCR/tree/v2.10.0)

https://github.com/PaddlePaddle/PaddleOCR/releases?page=2<br>
(docs\Releases-v2.10.0-PaddlePaddle-PaddleOCRv-github.com.mhtml)
<br>
<br>
將使用 PP-OCRv3 版本中的 [ch_PP-OCRv3_rec_slim](https://www.paddleocr.ai/v2.10.0/ppocr/model_list.html#21) 模型來開發車牌辨識<br>
ch_PP-OCRv3_rec_slim	slim量化版超轻量模型，支持中英文、数字识别	ch_PP-OCRv3_rec_distillation.yml	4.9M<br>
[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar)
[训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_train.tar)
<br>
<br>
<br>
<br>
參考來源:
模型庫 https://www.paddleocr.ai/v2.10.0/ppocr/model_list.html#21<br>
(docs\PaddleOCR-v2.10.0_PP-OCR_模型列表.mhtml)
<br>
<br>
<br>
<br>
## 環境設定
Python:3.9.25

### 1. 完全清理
> pip uninstall paddlepaddle paddleocr paddlehub numpy opencv-contrib-python opencv-python -y<br>
> pip cache purge

<br>

### 2. 安裝
> pip install -r requirements.txt

若是只使用CPU測試，可以省略，已安裝在 requirements.txt 裡面了
> pip install paddlepaddle

您的机器安装的是CUDA 11(GPU)，请运行以下命令安装
> pip install paddlepaddle-gpu

但是 requirements.txt 已經先安裝了 paddlepaddle，所以GPU版本的要重裝
> pip uninstall paddlepaddle-gpu paddlepaddle -y

#### 安裝支援 CUDA 11.8 的 2.6 版本(根據您的 CUDA 調整，假設您的 CUDA 是 11.x)
> python -m pip install paddlepaddle-gpu==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

#### CUDA 版本查詢<br>
> nvcc -V<br>
=>	<br>
	nvcc: NVIDIA (R) Cuda compiler driver<br>
	Copyright (c) 2005-2022 NVIDIA Corporation<br>
	Built on Wed_Sep_21_10:33:58_PDT_2022<br>
	Cuda compilation tools, release 11.8, V11.8.89<br>
	Build cuda_11.8.r11.8/compiler.31833905_0<br>

#### 驗證 paddlepaddle-gpu 安裝是否成功
> python -c "import paddle; paddle.utils.run_check()"<br>
=>	<br>
	Running verify PaddlePaddle program ...<br>
	I0107 08:31:00.438566 412716 program_interpreter.cc:212] New Executor is Running.<br>
	W0107 08:31:00.438954 412716 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 12.4, Runtime API Version: 11.8<br>
	W0107 08:31:00.468246 412716 gpu_resources.cc:164] device: 0, cuDNN Version: 8.9.<br>
	I0107 08:31:00.765264 412716 interpreter_util.cc:624] Standalone Executor is Used.<br>
	PaddlePaddle works well on 1 GPU.<br>
	PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.<br>

<br>

### 3. 查詢各套件版本
[python]<br>
python --version<br>
=>3.9.25<br>

[numpy]<br>
python -c "import numpy; print(numpy.__version__)"<br>
=>1.24.3<br>

[paddlepaddle]<br>
##### CPU WIN10
python -c "import paddle; print('paddle OK:', paddle.__version__)"<br>
=>3.0.0-rc1<br>

##### GPU Linux
python -c "import paddle; print('paddle OK:', paddle.__version__)"<br>
=>2.6.1<br>

[paddleocr]<br>
##### CPU WIN10
pip show paddleocr | findstr Version<br>
=>2.10.0<br>

##### GPU Linux
pip show paddleocr | grep Version<br>
=>2.10.0<br>

[opencv]<br>
python -c "import cv2; print('OpenCV 版本:', cv2.__version__)"<br>
##### CPU WIN10
=> 4.10.0<br>

##### GPU Linux
=>4.11.0<br>

<br>
<br>
<br>
<br>

## 使用(已整合到 .vscode\launch.json)

### 3種資料集
三種數據集的區別為了讓模型訓練更科學，通常我們會將數據分為以下三類：<br>
數據類型作用您的現狀訓練集 (Train Set)給模型「讀書」用的。模型會反覆看這些圖來學習特徵。train_list.txt <br>
驗證集 (Eval Set)給模型「模擬考」用的。模型不看答案，考考看學得如何。val_list.txt <br>
測試集 (Test Set)「真實戰場」。完全沒在清單內，直接拿一張新照片來辨識。非清單內的獨立圖片 <br>

### 模型分為 訓練(train)/推論(infer)
.pdparams：通常是訓練權重 (Student/Teacher Model) 的格式。(網絡結構)	预训练模型	動態圖模型<br>
.pdiparams：通常是推論模型 (Inference Model) 的權重格式。(權重參數)	訓練完成的模型	靜態圖模型<br>

Global.pretrained_model (預訓練/評估)： 它只載入權重（.pdparams）。當你進行「評估（Eval）」或「推理（Infer）」和導出 (Export)時，我們不需要優化器的資訊，只需要模型變換出的那些參數。<br>
<br>
<br>

### 訓練 tools\train.py
> python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml

是重頭訓練，還是續接(微調)取決於 .yml 的設定為何<br>
#### 重頭訓練
pretrained_model: ./pretrained/best_accuracy # 预训练模型位置，訓練權重，免加副檔名 [.pdparams]<br>
checkpoints:  # 重頭訓練，維持空值<br>

#### 續接(微調)
pretrained_model: # 續接時這裡可以留空<br>
checkpoints: ./output/rec_ppocr_v3_distillation/latest # 检测点文件位置，可通过设置此选项恢复训练<br>

訓練次數達到 .yml 設定檔中的 eval_batch_step 範圍次數就會去自動執行 eval.py<br>
讓最好的權重自動更新成 best_accuracy.pdparams

最後跑完 500迴圈的結果(根據 rec_carplate_train_gpu.yml 的設定 epoch_num: 500)
> 
[2026-01-07 16:01:41,507] ppocr INFO: best metric, acc: 0.12499984375019532, is_float16: False, norm_edit_dis: 0.4220245319931446, Teacher_acc: 0.24999968750039064, Teacher_norm_edit_dis: 0.44702450074318356, fps: 115.20755908972299, best_epoch: 500<br>
[2026-01-07 16:01:44,721] ppocr INFO: save model in ./output/rec_ppocr_v3_distillation/latest<br>
[2026-01-07 16:01:48,068] ppocr INFO: save model in ./output/rec_ppocr_v3_distillation/iter_epoch_500<br>
[2026-01-07 16:01:48,068] ppocr INFO: best metric, acc: 0.12499984375019532, is_float16: False, norm_edit_dis: 0.4220245319931446, Teacher_acc: 0.24999968750039064, Teacher_norm_edit_dis: 0.44702450074318356, fps: 115.20755908972299, best_epoch: 500<br>

數據解讀：模型現在的實力(訓練結果跑驗證)<br>
- acc: 0.1249 (12.5%) 這代表在你的 8 筆驗證資料中，只認對了 1 張。剩下的 7 張全都認錯了（OCR 的 Acc 要求是文字內容 100% 完全正確才算對）。<br>
- norm_edit_dis: 0.422 這是「正規化編輯距離」。數值越接近 1 越好。0.42 代表平均來說，一張車牌如果你有 7 個字，它可能只認對了 2~3 個字，或者順序完全亂掉。<br>
- Teacher_acc: 0.249 (25%) 老師模型（Teacher）認對了 2 張。這說明即便是結構更複雜的老師模型，目前的表現也很差。<br>


### 評估(也就是驗證) tools\eval.py
這是讓您在訓練結束後，或是想要針對某個特定的權重檔案（例如 iter_epoch_1000.pdparams）進行詳細檢查時手動使用的<br>

驗證模型的「真實實力」<br>
如果 acc > 0.9 即可導出模型使用了<br>
如果 acc 還是很低，但  norm_edit_dis 很高，代表模型認得出字，但容易混淆相似字（例如 8 和 B、0 和 D）。<br>
> python3 tools\eval.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/iter_epoch_1000

### 畫圖(Training Loss Over Global Steps)  draw_log.py
讀取訓練完成的 train.log 畫出其過程的 Loss 圖，方便觀察曲線圖<br>
(output\rec_ppocr_v3_distillation\training_loss_plot.png)

> ※ X 軸顯示的是「Step」，而不是「Epoch」(輪次)。

#### 總步數Total Steps(Global Steps)
所以，如果您的訓練完整跑完了 500 個 Epoch，X 軸理論上會一直延伸到 1000。如果您現在看到的圖表 X 軸尚未停在 1000，<br>
這代表：訓練尚未結束，可能因為報錯或是您手動停止。<br>
<br>
> ※ 假設您的 batch_size(yml設定檔中的 [train] batch_size_per_card=12) 是 12，總數據是 24，所以 1 epoch = 2 steps<br>

在深度學習中，公式如下：<br>
Total Steps = Epochs × (Total Samples / Batch Size)<br>
![Total Steps Formula](docs/total_steps.svg)

LaTeX公式:<br>
$$\text{Total Steps} = \text{Epochs} \times \left( \frac{\text{Total Samples}}{\text{Batch Size}} \right)$$
<br>
<br>

根據您的設定：<br>
- Total Samples（總數據量）: 24 筆。
- Batch Size（批大小）: 2 筆（batch_size_per_card: 12）。<br>
> 每輪步數: $24 \div 12 = 2$ Steps。<br>
- 總輪次 (Epochs): 500 次。

計算結果(總步數Total Steps)：
$500 \times 2 = \mathbf{1000}$ Steps。

<br>
<br>
<br>
<br>