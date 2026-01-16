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

[訓練集] 數據類型作用您的現狀訓練集 (Train Set)給模型「讀書」用的。(train_list.txt)<br>

>	模型會反覆看這些圖來學習特徵。<br>
	加入合成圖： 字體邊緣清晰、背景底色純淨、光影均勻讓模型看過各種排列組合，學會「字形」。<br>
	現實狀況：您的 180 張真實車牌中，可能剛好沒有出現字母 Q 或數字 7。如果模型從未見過這些字，它就永遠認不出來。<br>
	解決「字元覆蓋率」不足的問題。<br>
	強化模型對「字體結構」的理解。<br>
	能迫使模型專注於學習文字的筆畫結構。模型能更精準地分辨易混淆字元，例如 8 與 B、0 與 D、使 與 便。<br>
	如果推論時有認錯的字，便可透過合成資料來補強這些特定字元。<br>

	推薦的「黃金比例」:
	建議真實與合成的比例控制在 1 : 3 到 1 : 5 之間。
	真實照片 (Real)：180 張（其中約 150 張入訓練集，30張當作驗證集；約8:2或是7:3）。真實照片不夠可以用複製法。
	合成照片 (Synthetic)：約 500 ~ 800 張。
	總訓練量：約 700 ~ 1000 張。Epoch 跑 1000。

[評估/驗證集] (Eval Set)給模型「模擬考」用的。模型不看答案，考考看學得如何。(val_list.txt)<br>

>	評估/驗證集的作用是「照妖鏡」<br>
	評估集（val_list.txt）是用來告訴您：「模型現在在現實世界中表現如何？」<br>
	如果評估集裡有您那 24 張「軍外使」真實照片和一些一般車牌的真實截圖。<br>
	當訓練過程中 best_accuracy 提升時，您才能百分之百確定模型是真的學會認車牌了，而不是只學會認「合成軟體的字體」。<br>
	絕對不能出現在 train_list.txt 裡（這叫數據洩露，會導致評估失真）。<br>

[測試集] (Test Set)「真實戰場」，可以當作是評估/驗證集(可相同 val_list.txt)。<br>

>	完全沒在清單內，直接拿一張新照片來辨識。非清單內的獨立圖片，也是真實照片<br>

### 模型分為 訓練(train)/推論(infer)
.pdparams：通常是訓練權重 (Student/Teacher Model) 的格式。(網絡結構)	预训练模型	動態圖模型<br>
.pdiparams：通常是推論模型 (Inference Model) 的權重格式。(權重參數)	訓練完成的模型	靜態圖模型<br>

Global.pretrained_model (預訓練/評估)： 它只載入權重（.pdparams）。當你進行「評估（Eval）」或「推理（Infer）」和導出 (Export)時，我們不需要優化器的資訊，只需要模型變換出的那些參數。<br>
<br>
<br>

### 訓練 tools\train.py
> python tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml

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

驗證模型的「真實實力」(只能看訓練集數據，不能考試測試其他的圖片效果)<br>
如果 acc > 0.9 即可導出模型使用了<br>
如果 acc 還是很低，但  norm_edit_dis 很高，代表模型認得出字，但容易混淆相似字（例如 8 和 B、0 和 D）。<br>
> python tools\eval.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/iter_epoch_1000
<br>

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

### 推論，進行「真實預測」 (Inference) tools\infer_rec.py
使用訓練產出的權重 (.pdparams) 進行真實預測(用訓練模型預測)<br>
直接拿一張沒在訓練集裡的車牌照片，看看模型吐出什麼字<br>
必須拿切好的車牌(小圖)<br>
模擬訓練環境：儘量讓裁切的邊緣與訓練集相似(標籤時，訓練集都在訓練什麼的方式，一樣的規則拿來測試)<br>
如果您給它一張包含「整台車」甚至「街景」的照片，模型會因為雜訊太多（車燈、輪胎、樹木）而完全無法辨識<br>

> python tools/infer_rec.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=./output/rec_carplate/best_accuracy Global.infer_img=要測試的圖片路徑.jpg

> 
	profiler_options : None
	[2026/01/14 17:12:36] ppocr INFO: train with paddle 2.6.1 and device Place(gpu:0)
	W0114 17:12:36.310602 2434868 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 12.4, Runtime API Version: 11.8
	W0114 17:12:36.311692 2434868 gpu_resources.cc:164] device: 0, cuDNN Version: 8.9.
	[2026/01/14 17:12:36] ppocr INFO: load pretrain successful from ./output/rec_ppocr_v3_distillation/best_accuracy
	[2026/01/14 17:12:36] ppocr INFO: infer_img: ./no_train_images/no_train_img1.jpg
	[2026/01/14 17:12:37] ppocr INFO:        result: {"Student": {"label": "外2167", "score": 0.5458627939224243}, "Teacher": {"label": "外167", "score": 0.501935601234436}}
	[2026/01/14 17:12:37] ppocr INFO: success!

### 導出推理模型(Export) tools\export_model.py
將訓練好的動態圖導出為靜態圖推理模型，這樣之後可以用更快的速度部署(訓練模型導出推論模型)<br>
PaddleOCR 的推理（Inference）流程需要一個專門的、經過模型結構剪枝和優化的格式，通常由兩個檔案組成：<br>
model.pdmodel：僅包含模型的網路結構。<br>
model.pdiparams：僅包含最終的權重參數。<br>

訓練好的 best_accuracy.pdparams 雖然有權重，但它缺乏優化後的結構資訊，且包含了訓練時的額外層次。<br>

因此要轉換為推理模型（Exporting Inference Model）<br>
將您的訓練成果用於實際的車牌識別，您必須將儲存在 best_accuracy.pdparams 中的權重轉換（Export）成 PaddleOCR 可讀取的推理格式。<br>

> python tools/export_model.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml o Global.pretrained_model=./output/rec_carplate/best_accuracy Global.save_inference_dir=./inference/rec_v3 Global.model_type=rec

>	
	[2026/01/16 14:44:17] ppocr WARNING: Skipping import of the encryption module.
	W0116 14:44:17.807654 3758036 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 12.4, Runtime API Version: 11.8
	W0116 14:44:17.808665 3758036 gpu_resources.cc:164] device: 0, cuDNN Version: 8.9.
	[2026/01/16 14:44:18] ppocr INFO: load pretrain successful from ./output/rec_ppocr_v3_distillation/best_accuracy
	[2026/01/16 14:44:18] ppocr INFO: Export inference config file to ./inference/PP-OCRv3_taiwan_car_rec/inference.yml
	Skipping import of the encryption module
	I0116 14:44:20.362078 3758036 program_interpreter.cc:212] New Executor is Running.
	[2026/01/16 14:44:20] ppocr INFO: inference model is saved to ./inference/PP-OCRv3_taiwan_car_rec/Teacher/inference
	Skipping import of the encryption module
	[2026/01/16 14:44:21] ppocr INFO: inference model is saved to ./inference/PP-OCRv3_taiwan_car_rec/Student/inference

您的模型轉換是成功的，這三個檔案就是您需要的推理模型！<br>
inference.pdiparams：經過優化、用於推理的最終權重參數。<br>
inference.json 或 inference.yml 則包含了模型運行所需的結構資訊或元數據。<br>

導出後你會得到三個檔案：2.x<br>
inference.pdmodel（模型結構）<br>
inference.pdiparams（模型權重)<br>
inference.pdiparams.info（額外資訊）<br>

新版本的 3.x<br>
inference.json（模型結構）<br>
inference.pdiparams（模型權重)<br>
inference.yml（額外資訊）<br>
<br>

導出的的推論模型存檔路徑(Global.save_inference_dir)會有兩個資料夾(Student 和 Teacher)<br>
直接使用 Student 資料夾中的權重。<br>
然後手動刪除 Teacher 資料夾即可。<br>

為什麼會有這兩個資料夾？<br>
PaddleOCR v3 的識別模型使用的是 DML (Deep Mutual Learning) 蒸餾策略。<br>

Teacher (老師模型)： 在 PP-OCRv3 中，老師模型通常是一個結構較大、預測能力較強的模型，它的任務是引導學生模型學習。<br>
Student (學生模型)： 這是經過優化的輕量化模型（例如 MobileNetV3 結構）。它的目標是達到接近老師的準確率，但同時保持極快的推論速度。<br>

為什麼要選 Student？
部署效率： Student 模型才是真正設計來進行部署的「輕量化版本」。它的推論速度最快，佔用的記憶體與 GPU 資源最少。<br>
準確率： 經過 DML 訓練後，學生模型的表現通常已經非常接近甚至達到老師的水準。<br>
相容性： 大部分 PaddleOCR 提供的預測腳本（如 predict_rec.py）預設都是對應 Student 結構。<br>

### 預測(Prediction) tools\infer\predict_rec.py
導出後就可以直接使用 PaddleOCR API 進行推理。(用推論模型預測)<br>
導出後的模型，進行測試<br>

> python tools/infer/predict_rec.py --image_dir="要測試的圖片檔路徑" --rec_model_dir="你的推論模型路徑" --rec_image_shape="3, 48, 320" --rec_char_dict_path="你的字典路徑" --use_gpu="False"

>	
	(py309OCRppv3) F:\UserData\Larry\Documents\VSCode Project\Python\PaddleOCR-2.10.0> f: && cd "f:\UserData\Larry\Documents\VSCode Project\Python\PaddleOCR-2.10.0" && cmd /C ""d:\Program Files\anaconda3\envs\py309OCRppv3\python.exe" c:\Users\Larry\.vscode\extensions\ms-python.debugpy-2024.8.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher 11974 -- "F:\UserData\Larry\Documents\VSCode Project\Python\PaddleOCR-2.10.0/tools/infer/predict_rec.py" --image_dir ./no_train_images/no_train_img1.jpg --rec_model_dir ./inference/PP-OCRv3_taiwan_car_rec/Student --rec_image_shape "3, 48, 320" --rec_char_dict_path ./ppocr/utils/dict/dict_taiwan_car.txt --use_gpu False "
	INFO: Could not find files for the given pattern(s).
	d:\Program Files\anaconda3\envs\py309OCRppv3\lib\site-packages\paddle\utils\cpp_extension\extension_utils.py:711: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
	  warnings.warn(warning_message)
	[2026/01/16 15:32:28] ppocr INFO: In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320
	[2026/01/16 15:32:28] ppocr INFO: Predicts of ./no_train_images/no_train_img1.jpg:('外2167', 0.5458630323410034)

	(py309OCRppv3) F:\UserData\Larry\Documents\VSCode Project\Python\PaddleOCR-2.10.0>
<br>
<br>
<br>
<br>