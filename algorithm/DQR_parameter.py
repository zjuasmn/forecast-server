# coding=UTF-8
# ----------------------------------- #
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
# ----------------------------------- #

# 根据预测对象与预测尺度设置DQR预测相关参数
def get_para(type, scale):

  # DQR输入层神经元数目等于输入变量数
  if type == "PHOTOVOLTAIC" and scale == "ultrashort":
    # 待预测时刻前7个时刻功率值、前7天相同时刻功率
    in_num = 7+7
  elif type == "PHOTOVOLTAIC" and scale == "short":
    # 前7天相同时刻功率
    in_num = 7
  elif type == "LOAD" and scale == "ultrashort":
    # 待预测时刻前7个时刻功率值，前7天相同时刻功率值
    in_num = 7+7
  elif type == "LOAD" and scale == "short":
    # 前7天相同时刻功率值
    in_num = 7

  # 超短期预测训练时需要训练16个模型；短期预测训练时需要训练1个模型
  if scale == "ultrashort":
    point_num = 16
  elif scale == "short":
    point_num = 1

  return in_num, point_num