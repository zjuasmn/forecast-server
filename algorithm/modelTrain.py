# ----------------------------------- #
import json
import copy
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
from DQR_model import DQR
from DQR_parameter import *
from DQR_dataset import *
from DQR_evaluation import performance
import sys
# ----------------------------------- #


def train(time, data, entity):
  
  # 从model.json读取model，如果model.json文件不存在或文件为空，则创建新的model变量
  tf = str(entity['type']) + '-' + str(entity['forecastScale'])
  try:
    with open('./model.json', encoding='utf8') as modelFile:
      model = json.load(modelFile)
    model[tf] = {} # 初始化model变量
  except IOError:
    model = {}
    model[tf] = {}
  except json.decoder.JSONDecodeError:
    model = {}
    model[tf] = {}
  
  # 根据预测对象、预测尺度设定ELM网络的输入神经元数与预测时刻数
  in_num, point_num = get_para(entity["type"], entity["forecastScale"])
  # 设定ELM网络的输出神经元数以及分位数
  out_num = 1
  quantiles = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

  for pointnum in range(point_num): # 逐个训练预测模型，超短期需要训练16个模型，短期需要训练1个
    
    print("pointnum = ", pointnum)

    # 数据需整理格式，三个数据集分别代表全部数据、训练数据、测试数据
    all_data, train_data, test_data = dataset_form_train(data = data, \
      entity = entity, horizon = pointnum)
    print("all_data: ", np.shape(all_data))  
    # 将数据集分割成输入与输出
    X, Y = all_data[:, :in_num], all_data[:, in_num:]
    X_train, Y_train = train_data[:, :in_num], train_data[:, in_num:]
    X_test, Y_test = test_data[:, :in_num], test_data[:, in_num:]

    pfDict = {} # 存放参数寻优的各个参数组合的指标
    count = 0   
    # ----------------------------------------------------
    for hidnum in range(10, 20, 10): # 参数寻优确定hid_num
    # ----------------------------------------------------
      print("hidnum = ", hidnum)
        
      print("model training")
      # 采用训练集训练模型
      dqr = DQR(quantiles = quantiles, n_feature = in_num, n_hidden = hidnum, n_output = out_num)
      dqr.fit(X_train, Y_train)

      print("model testing")
      # 采用测试集测试模型
      y = dqr.predict(X_test)

      mae, acd, sc = performance(Y_test, y, quantiles)
      # mape_acd = 0.2 * (1 - mape) / 100 + acd * 0.8  # 计算一个综合指标

      print("test ouput storing")
      # 将寻优参数和对应指标存进pfDict
      pfDict[count] = {
        "para": hidnum, 
        "mae": mae, 
        "acd": acd, 
        "sc": sc
      }
      count += 1 
    
    print("searching for best parameter")
    # 寻找最优参数组合
    pfDf = pd.DataFrame(pfDict.values())
    # --------------
    # best_para = pfDf["para"][pfDf["acd"].idxmax()]
    best_para = pfDf["para"][pfDf["sc"].idxmax()]
    # --------------
    print(pfDf)
    print('point{0}-hidnum: {1}'.format(pointnum, best_para))
    best_hidnum = int(best_para)

    # 将需要调优的参数放进包里
    DQR_para_dict = {
      "in_num": in_num,
      "hid_num": best_hidnum,
      "out_num": out_num 
    }

    print("model training with best parameter")
    # 采用全部数据集训练模型
    # 训练模型，得到模型参数
    dqr = DQR(quantiles = quantiles, n_feature = in_num, n_hidden = hidnum, n_output = out_num)
    dqr.fit(X, Y)

    print("model parameter packaging")
    # 参数打包
    model[str(entity["type"]) + '-' + str(entity["forecastScale"])]["quantiles"] = copy.deepcopy(quantiles).tolist()
    model[str(entity["type"]) + '-' + str(entity["forecastScale"])]["point_num"] = point_num
    model[str(entity["type"]) + '-' + str(entity["forecastScale"])]["point" + str(pointnum)] = {
      'DQR_para_dict': copy.deepcopy(DQR_para_dict),
      'in_weight': copy.deepcopy(dqr.w).tolist(),
      'in_bias': copy.deepcopy(dqr.b).tolist(),
      'out_weight': copy.deepcopy(dqr.W).tolist()
    }
  return model


if __name__ == '__main__':
  if len(sys.argv) != 3:
    print('usage: modelTrain.py [trainInput.json] [modelOutput.json]')
    exit(1)
  with open(sys.argv[1], encoding='utf-8') as trainInputFile:
    trainInput = json.load(trainInputFile)
  time = trainInput["time"]
  entity = trainInput["entity"]
  data = trainInput["data"]
  model = train(time, data, entity)
  # 将更新后的model存放到model.json文件中
  with open(sys.argv[2], encoding = 'utf8', mode = 'w') as modelFile:
    json.dump(model, modelFile, indent = 2)
  
  
  
