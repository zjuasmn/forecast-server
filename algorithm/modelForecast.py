# ----------------------------------- #
from DQR_model import DQR
from DQR_dataset import *
import sys
# ----------------------------------- #

def forecast(model, power, entity):

  tf = str(entity['type']) + '-' + str(entity['forecastScale'])
  id = entity['id']


  model = model[tf]               # 取对应type和forecastScale的model     
  quantiles = np.array(model['quantiles'])  # 概率预测分位数
  point_num = model['point_num']  # 预测点数

  # 从refValue.json读取基准值
  with open('./refValue.json', encoding='utf8') as refValueFile:
    refValue = json.load(refValueFile)
  powerMaxValue = refValue[tf][id]

  if entity['forecastScale'] == 'ultrashort':
    Y = np.zeros((quantiles.shape[0], point_num))  # shape: (n_quantile, 16)
  elif entity['forecastScale'] == 'short':
    Y = np.zeros((quantiles.shape[0], point_num*96))  # shape: (n_quantile, 96)

  for pointnum in range(point_num):
   
    # 参数解包 
    DQR_para_dict = model["point" + str(pointnum)]['DQR_para_dict']
    in_num = DQR_para_dict['in_num']
    hid_num = DQR_para_dict['hid_num']
    out_num = DQR_para_dict['out_num']
    w = np.array(model["point" + str(pointnum)]['in_weight'])
    b = np.array(model["point" + str(pointnum)]['in_bias'])
    W = np.array(model["point" + str(pointnum)]['out_weight'])

    # 整理数据格式
    X = dataset_form_forecast(power = power, entity = entity, \
      horizon = pointnum, powerMaxValue = powerMaxValue)

    # 预测
    dqr = DQR(quantiles = quantiles, n_feature = in_num, n_hidden = hid_num, n_output = out_num)
    subY = dqr.predict(X, quantiles = quantiles, w = w, b = b, W = W)  # shape: (n_sample, n_quantile)

    if entity['forecastScale'] == 'ultrashort':
      Y[:, pointnum] = subY
    elif entity['forecastScale'] == 'short':
      Y = subY.T

  Y *= powerMaxValue  # 反归一化：Y*基准值
  # 将结果转换为dataframe，最后转换为dict
  forecastOutput = pd.DataFrame(Y, index = quantiles)
  forecastOutput = forecastOutput.to_dict(orient = 'dict')
  return forecastOutput


if __name__ == '__main__':
  if len(sys.argv) != 4:
    print('usage: modelForecast.py [forecastInput.json] [model.json] [forecastOutput.json]')
    exit(1)
  with open(sys.argv[1], encoding='utf-8') as forecastInputFile:
    forecastInput = json.load(forecastInputFile)
  time = forecastInput["time"]
  entity = forecastInput["entity"]
  power = forecastInput["power"]
  # 从model.json读取model
  with open(sys.argv[2], encoding = 'utf8') as modelFile:
    model = json.load(modelFile)
  forecastOutput = forecast(model, power, entity)
  # 将预测结果写入forecastOutput.json文件
  with open(sys.argv[3], encoding = 'utf8', mode = 'w') as forecastOutputFile:
    json.dump(forecastOutput, forecastOutputFile, indent = 2)

  # tf = str(entity['type']) + '-' + str(entity['forecastScale'])
  # id = entity['id']
  # with open('./model.json', encoding='utf8') as modelFile:
  #   model = json.load(modelFile)
  # model = model[tf]               # 取对应type和forecastScale的model
  # quantiles = np.array(model['quantiles'])  # 概率预测分位数
  # point_num = model['point_num']  # 预测点数

  # # 从refValue.json读取基准值
  # with open('./refValue.json', encoding='utf8') as refValueFile:
  #   refValue = json.load(refValueFile)
  # powerMaxValue = refValue[id]

  # with open('./forecastOutput.json', encoding='utf8') as forecastOutputFile:
  #   json_data = json.load(forecastOutputFile)
  # powerOutput = pd.DataFrame(json_data).values.T
  # powerOutput = pd.DataFrame(json_data).values.T / powerMaxValue

  # dataSource = pd.read_excel('./dataSourceTest.xlsx', encoding = 'utf8', index_col = [0])
  # truePower = dataSource.iloc[365*96:365*96+16, :].loc[:, 'LA1'].round(4).values.reshape(-1, ) / powerMaxValue
  # mae, acd, sc = performance(truePower, powerOutput, quantiles)
  # print(mae, acd, sc)

  # plt.figure()
  # plt.plot(truePower, c='black')
  # plt.plot(powerOutput, c='blue')
  # plt.show()


