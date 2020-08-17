# coding=UTF-8
# ----------------------------------- #
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
# ----------------------------------- #

def dataset_form_train(data, entity, horizon, day_num=96, past_day=7, past_time=7):
  """
  将用来训练DQR模型的数据整理成特定矩阵格式[input|output]
  光伏超短期数据格式：前past_day天相同时刻，前past_time个时刻，输出
  光伏短期数据格式：前past_day天相同时刻，输出
  负荷超短期数据格式：前past_day天相同时刻，前past_time个时刻，输出
  负荷短期数据格式：前past_day天相同时刻，输出
  :param data: 功率数据
  :param entity: 预测实体，需要用到type和forecastScale参数
  :param horizon: 超短期预测涉及预测未来的第1个至第16个时刻点，不同时刻点的训练数据不一样
  :param day_num: 数据分辨率，一天96个点
  :param past_day: 表示取前past_day天相同时刻的功率数据
  :param past_time: 表示取前past_time个时刻的功率数据
  :return: 格式化后的数据矩阵, shape: (n_sample, n_feature + n_output)
  """

  # 从refValue.json读取归一化基准值，如果refValue.json文件不存在或文件为空，则创建新的refValue变量
  tf = str(entity['type']) + '-' + str(entity['forecastScale'])
  try:
    with open('./refValue.json', encoding='utf8') as refValueFile:
      refValue = json.load(refValueFile)
    refValue[tf] = {}
  except IOError:
    refValue = {}
    refValue[tf] = {}
  except json.decoder.JSONDecodeError:
    refValue = {}
    refValue[tf] = {}

  # 找出每个用户的功率数据最大值，取其mul倍作为基准值存进refValue中
  mul = 1.5
  for id in list(data.keys()):
    subPower = data[id]["power"]
    powerMaxValue = round(mul * max(subPower), 4)
    refValue[tf][id] = powerMaxValue

  # 将refValue存放到refValue.json文件中
  with open('./refValue.json', encoding='utf8', mode='w') as refValueFile:
    json_data = json.dump(refValue, refValueFile, indent=2)

  # 将每个用户的数据单独处理成需要的格式，后续会拼合所有用户的数据
  allDataSet = []  # 存放单个用户的数据
  for id in data.keys():
    subData = data[id]  # 取出单个用户的数据
    subPower = subData["power"] # 单个用户的功率数据

    # 归一化
    powerMaxValue = refValue[tf][id]  # 根据用户id从refValue中索引基准值
    subPower = (np.array(subPower) / powerMaxValue).tolist()   # 功率数据归一化
    type, scale = entity["type"], entity["forecastScale"] # 预测对象类型、预测尺度

    if type == "LOAD" and scale == "ultrashort":
      set_1 = get_set1(subPower, day_num, past_day)   # 前几天相同时刻
      set_2 = get_set2(subPower, horizon, day_num, past_day, past_time)  # 前几个时刻
      set_4 = np.array([subPower[day_num * past_day:].copy()]).T  # 输出
      set = np.hstack((set_1, set_2, set_4))
    elif type == "LOAD" and scale == "short":
      set_1 = get_set1(subPower, day_num, past_day)   # 前几天相同时刻
      set_4 = np.array([subPower[day_num * past_day:].copy()]).T  # 输出
      set = np.hstack((set_1, set_4))
    elif type == "PHOTOVOLTAIC" and scale == "ultrashort":
      set_1 = get_set1(subPower, day_num, past_day)   # 前几天相同时刻
      set_2 = get_set2(subPower, horizon, day_num, past_day, past_time)  # 前几个时刻
      set_4 = np.array([subPower[day_num * past_day:].copy()]).T  # 输出
      set = np.hstack((set_1, set_2, set_4))
    elif type == "PHOTOVOLTAIC" and scale == "short":
      set_1 = get_set1(subPower, day_num, past_day)   # 前几天相同时刻
      set_4 = np.array([subPower[day_num * past_day:].copy()]).T  # 输出
      set = np.hstack((set_1, set_4))

    allDataSet.append(set)

  # 将所有用户的数据拼合在一起
  set = np.vstack(allDataSet)

  # 随机分成训练集与测试集，大小为9:1、
  # 由于数据集太大，随机选取18000条、2000条样本构建训练集和测试集
  row_rand_array = np.arange(set.shape[0])
  np.random.shuffle(row_rand_array)
  # ----------------------
  # train_set = set[row_rand_array[0:int(set.shape[0] * 0.9)]]
  # test_set = set[row_rand_array[int(set.shape[0] * 0.9):]]
  # set = np.append(train_set, test_set, axis=0)
  train_set = set[row_rand_array[:180]]
  test_set = set[row_rand_array[180:200]]
  set = set[row_rand_array[:200]]
  # -----------------------

  return set, train_set, test_set

# 前几天相同时刻数据格式化, 矩阵最左边表示前一天，最右边表示往前第7天
def get_set1(power, day_num, past_day):
  set_1 = []
  for i in range(past_day):
      set_1.append(power[day_num * (past_day - i - 1):-day_num * (i + 1)].copy())
  set_1 = np.array(set_1).T
  return set_1

# 前几个时刻数据格式化, 矩阵最左边表示前一个时刻，最右边表示往前第7个时刻
def get_set2(power, horizon, day_num, past_day, past_time):
  set_2 = []
  for i in range(past_time):
    set_2.append(power[day_num * past_day - horizon - past_time + i: - horizon - past_time + i].copy())
  set_2 = np.array(set_2).T
  return set_2

def dataset_form_forecast(power, entity, horizon, powerMaxValue, day_num=96, past_day=7, past_time=7):
  """
  将用DQR模型预测的数据整理成特定矩阵格式[input]
  光伏超短期输入数据格式：前past_day天相同时刻，前past_time个时刻
  光伏短期输入数据格式：前past_day天相同时刻
  负荷超短期输入数据格式：前past_day天相同时刻，前past_time个时刻
  负荷短期输入数据格式：前past_day天相同时刻
  :param power: 功率数据序列
  :param entity: 预测实体，需要用到type和forecastScale参数
  :param horizon: 超短期预测涉及预测未来的第1个至第16个时刻点，不同时刻点用到的前几天相同时刻不一样
  :param powerMaxvalue: 功率数据归一化基准值
  :param day_num: 数据分辨率，一天96个点
  :param past_day: 表示取前past_day天相同时刻的功率数据
  :param past_time: 表示取前past_time个时刻的功率数据
  :return: 格式化后的数据矩阵, shape: (n_sample, n_feature)
  """
  # -----------------
  power = (np.array(power) / powerMaxValue).tolist()    # 功率数据归一化
  # power = (np.array(power)).tolist()
  # -----------------
  type, scale = entity["type"], entity["forecastScale"] # 预测对象类型、预测尺度

  if type == "LOAD" and scale == "ultrashort":
    set_1 = get_set1_forecast(power, day_num, past_day, horizon)   # 待预测时刻前几天相同时刻
    set_2 = get_set2_forecast(power, past_time)  # 待预测时刻前几个时刻
    set = np.hstack((set_1, set_2))
  elif type == "LOAD" and scale == "short":
    # 待预测时刻前几天相同时刻，96个点的都需要, 矩阵最左边表示前一天，最右边表示往前第7天
    set_1 = []
    set_1.append(power[(-day_num):].copy())
    for i in range(1, past_day):
        set_1.append(power[(-day_num * (i + 1)):(-day_num * i)].copy())
    set = np.array(set_1).T
  elif type == "PHOTOVOLTAIC" and scale == "ultrashort":
    set_1 = get_set1_forecast(power, day_num, past_day, horizon)   # 待预测时刻前几天相同时刻
    set_2 = get_set2_forecast(power, past_time)  # 待预测时刻前几个时刻
    set = np.hstack((set_1, set_2))
  elif type == "PHOTOVOLTAIC" and scale == "short":
    # 待预测时刻前几天相同时刻，96个点的都需要, 矩阵最左边表示前一天，最右边表示往前第7天
    set_1 = []
    set_1.append(power[(-day_num):].copy())
    for i in range(1, past_day):
        set_1.append(power[(-day_num * (i + 1)):(-day_num * i)].copy())
    set = np.array(set_1).T
  return set

# 待预测时刻前几天相同时刻格式化, 矩阵最左边表示前一天，最右边表示往前第7天
def get_set1_forecast(power, day_num, past_day, horizon):
  set_1 = []
  for i in range(past_day):
    set_1.append(power[-day_num * i + horizon])
  set_1 = np.array([set_1])
  return set_1

# 待预测时刻前几个时刻数据格式化, 矩阵最左边表示前一个时刻，最右边表示往前第7个时刻
def get_set2_forecast(power, past_time):
  set_2 = []
  set_2.append(power[-1:(-past_time-1):-1])
  set_2 = np.array(set_2)
  return set_2

