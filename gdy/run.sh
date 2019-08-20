#!/bin/bash
#1.创建目录
mkdir stacking
mkdir preprocess_data

#2.预处理
python preprocess.py

#3.提取特征
python extract_features.py

#3.转换NN数据格式`
python convert_NN.py

#4.NN model
python NN.py

#5.LGB model
python LGB.py