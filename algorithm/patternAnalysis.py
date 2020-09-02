# ----------------------------------- #
import os
import json
import copy
import time
import math
import calendar
import numpy as np
np.seterr(divide = 'ignore', invalid = 'ignore')
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering
from evaluationIndicators import davies_bouldin_score, davies_bouldin_score_eu_cos
from sklearn.preprocessing import LabelEncoder
# ----------------------------------- #


def dataPreprocess(month, data):
    """
    将输入数据整理成算法运行所需格式
    """

    print('data preprocessing')
    # 根据所输入月份获取对应日期序列
    givenDate = datetime.strptime(month, "%Y-%m")  # 将month从str格式转换为datetime格式
    start_date = givenDate.replace(day=1)  # 月份首日日期
    _, days_in_month = calendar.monthrange(givenDate.year, givenDate.month)  # 月份天数
    end_date = givenDate + timedelta(days=days_in_month-1)  # 月份尾日日期
    dateseries = pd.date_range(start=start_date, end = end_date, freq = '1D')  # 当月日期序列
    timeseries = pd.date_range(start='20200101000000', end = '20200101234500', freq = '15min').strftime('%H-%M-%S')  # 每日采样时刻序列
    weekList = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']  # 聚类标签需按照周日至周一排序

    # 遍历输入的data中所有用户的id、power数据，将其整理成聚类输入所需格式
    powerDfList = []  # 存放每个用户的整理后的power数据
    # 遍历每个用户的数据
    for unit in data:
        power = np.array(unit['power']).reshape(len(dateseries), len(timeseries))  # 将原始数据转换为天数*时刻数的矩阵
        powerDf = pd.DataFrame(power, index = dateseries, columns = timeseries)  # 矩阵→dataframe
        powerDf['week'] = powerDf.index.weekday_name  # 获取每天对应星期几
        weekPowerDf = powerDf.groupby('week').mean()  # 对分属周日/周一/…/周六的日期进行求平均
        weekPowerDf = weekPowerDf.loc[weekList, :]  # 按照weekList给出的顺序进行排序
        weekPowerDf.index = [[unit['id']]*7, weekPowerDf.index]
        weekPowerDf.index.names = ['id', 'week']
        powerDfList.append(weekPowerDf)  # 将每个用户的整理后的power数据存进powerDfList
    powerDf = pd.concat(powerDfList, axis = 0)  # 将所有用户的数据进行拼合
    powerDf = powerDf.apply(lambda x: x / max(x), axis = 1)  # 将数据进行每一行进行最大值归一化

    return powerDf

def patternAnalysis(month, data):
    """
    能源消费特征分析算法主体
    """

    # 算法运行所需数据准备
    dataCluDf = dataPreprocess(month, data)  # 将data整理成聚类输入所需格式
    data_c = dataCluDf.values  # 从dataframe中获取矩阵形式的数据
    rows, cols = np.shape(data_c)  # 获取data_c行列数
    data_delta = np.diff(data_c, n = 1, axis = 1)  # 计算data_c的一阶差分向量, 用于以差分余弦距离作为距离度量的聚类

    # 算法运行所需参数设置
    k_min, k_max, k_step = 3, 10, 1   # 设置聚类簇数k取值范围及步长
    g_min, g_max, g_step = 0.05, 2.05, 0.1   # 设置gamma取值范围及步长
    c_eu, c_cos = 0.5, 0.5  # 设置综合距离计算时欧氏距离与差分余弦距离的权重

    # 算法运行参数寻优所需设置
    evaluationDict = {}   # 存放参数寻优的各个参数组合的指标
    paraCount = 0  # 参数组合的计数

    print('similarity distance calculating')
    # 算法运行所需距离矩阵计算
    distance_e = pairwise_distances(data_c, metric = 'euclidean')   # 计算data_c的欧氏距离矩阵
    distance_c = pairwise_distances(data_delta, metric = 'cosine')  # 计算data_delta的余弦距离矩阵
    eu_max, eu_min = np.max(distance_e), np.min(distance_e)
    cos_max, cos_min = np.max(distance_c), np.min(distance_c)
    scaling_ratio = (eu_max - eu_min) / (cos_max - cos_min)  # 计算两个距离矩阵的缩放比例
    distance_en = c_eu * distance_e + c_cos * distance_c * scaling_ratio   # 计算综合距离矩阵

    print('going through all parameter combinations')
    # 遍历所设置的k值，参数寻优
    for k in np.arange(k_min, k_max, k_step):
        # 遍历所设置的g值，参数寻优
        for g in np.arange(g_min, g_max, g_step):

            g = round(g, 2)

            # 使用每个参数组合进行一次聚类
            sp_cluster = SpectralClustering(n_clusters=k, eigen_solver=None, random_state = None, n_init = 30, \
                affinity = 'precomputed', gamma = 1, assign_labels = 'kmeans')
            affinity_matrix_c = np.exp(-g * distance_en)  # 计算谱聚类所需的相似矩阵
            label_sp = sp_cluster.fit_predict(affinity_matrix_c)  # 聚类得到标签
            dbiec = davies_bouldin_score_eu_cos(data_c, data_delta, label_sp)  # 计算该参数组合聚类结果的评价指标

            # 将每次参数组合和对应指标按顺序存进performance_dict
            evaluationDict[paraCount] = {
            "para":[k, g],  # 保存聚类类簇数与gamma值的组合
            "dbiec": dbiec,     # 保存该参数组合对应的评价指标结果
            }
            paraCount += 1

    print('searching for the best parameter combination')
    # 寻找最优参数组合
    evaluationDf = pd.DataFrame(evaluationDict.values())
    evaluationDf.set_index('para', inplace = True)
    best_para = evaluationDf["dbiec"].idxmin()
    print(evaluationDf)
    print(month, ': ', best_para, evaluationDf["dbiec"].min())
    k, g = best_para[0], best_para[1]

    print('clustering with the best parameter combination')
    # 依据最优参数组合进行模式分析
    sp_cluster = SpectralClustering(n_clusters=k, eigen_solver=None, random_state = None, n_init = 30, \
        affinity = 'precomputed', gamma = 1, assign_labels = 'kmeans')
    affinity_matrix_c = np.exp(-g * distance_en)  # 计算谱聚类所需的相似矩阵
    label_sp = sp_cluster.fit_predict(affinity_matrix_c)  # 聚类得到标签
    dataCluDf['label'] = label_sp + 1  # 将label_sp的最小值从0改为1

    # 将标签保存成为json文件
    print('storing clustering output')
    clusterOutput = {}
    clusterOutput['month'] = month
    for unit in data:
        del unit['power']
        unit['clusterByDay'] = dataCluDf.loc[unit['id'], 'label'].values.reshape(-1,).tolist()
    clusterOutput['data'] = data
    return clusterOutput
