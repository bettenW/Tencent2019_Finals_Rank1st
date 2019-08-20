#!/bin/bash
#1.创建目录
mkdir cache

#2.预处理
python preprocess_data.py
##
python gen_sub.py