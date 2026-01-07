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

### 訓練 tools\train.py
> python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy

<br>
<br>
<br>
<br>