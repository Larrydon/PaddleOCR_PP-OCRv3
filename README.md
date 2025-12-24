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

## 1. 完全清理
> pip uninstall paddlepaddle paddleocr paddlehub numpy opencv-contrib-python opencv-python -y

> pip cache purge

## 2. 安裝
> pip install -r requirements.txt

您的机器安装的是CUDA 11，请运行以下命令安装
> pip install paddlepaddle-gpu

若是只使用CPU測試，可以省略，已安裝在 requirements.txt 裡面了
> pip install paddlepaddle

## 3. 查詢各套件版本
[python]<br>
python --version<br>
=>3.9.25

[numpy]<br>
python -c "import numpy; print(numpy.__version__)"<br>
=>1.24.3

[paddlepaddle]<br>
python -c "import paddle; print('paddle OK:', paddle.__version__)"<br>
=>3.0.0-rc1

[paddleocr]<br>
pip show paddleocr | findstr Version<br>
=>2.10.0

[opencv]<br>
python -c "import cv2; print('OpenCV 版本:', cv2.__version__)"<br>
=> 4.10.0