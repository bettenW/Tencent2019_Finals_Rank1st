#!/bin/bash
#1.创建目录
mkdir preprocess_data

#2.预处理
python wh_preprocess.py

#3.特征加模型
python wh_LGB.py