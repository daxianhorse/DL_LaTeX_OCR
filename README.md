# DL_LaTeX_OCR
*神经网络与深度学习大作业*

## 环境说明
Anaconda latest\
tensorflow 2.7.0

## 目录结构说明
### 根目录
predict.py用作预测\
train.py用作训练\
evaluate.py用于计算edit_distance和exact_match\
app.py用于Web界面的后端

### data
数据目录\
biology:生物
maths:数学

### model
模型目前包括\
transformer\
rnn-mobienet\
rnn-efficenet\
rnn-efficenet-attention

build_dataset.py用于生成tf数据集

### weights
权重目录

### utils
组件库：图片切割，图片预处理，生成textvectorization

### reference
参考文献

### templates
UI界面设计（Web）

### train_history
训练历史记录 .txt

