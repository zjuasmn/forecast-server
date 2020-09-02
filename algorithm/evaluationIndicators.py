"""Unsupervised evaluation metrics."""

# Authors: Robert Layton <robertlayton@gmail.com>
#          Arnaud Fouchet <foucheta@gmail.com>
#          Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

from __future__ import division

import functools
import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import safe_indexing
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import time
import math


def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels

    n_samples : int
        Number of samples
    """
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)


def davies_bouldin_score(X, labels, metric = 'euclidean'):

    X, labels = check_X_y(X, labels) # 检查X和labels的尺寸
    le = LabelEncoder()
    labels = le.fit_transform(labels) # 将labels转换为0-k之间的整数
    n_samples, _ = X.shape      # 样本数
    n_labels = len(le.classes_) # label数
    check_number_of_labels(n_labels, n_samples) # 检查label数量是否正确

    intra_dists = np.zeros(n_labels) # 初始化类内距离
    centroids = np.zeros((n_labels, len(X[0])), dtype=np.float) # 初始化聚类中心
    for k in range(n_labels):
        cluster_k = safe_indexing(X, labels == k) # 索引样本集中标签为k的样本(即第k类)
        centroid = cluster_k.mean(axis=0)         # 求第k类样本的中心点
        centroids[k] = centroid
        # 求第k类样本到第k类中心的距离的平均值(以metric度量)
        intra_dists[k] = np.average(pairwise_distances(
            cluster_k, [centroid], metric = metric))

    # 求不同类中心的距离(以metric度量)
    centroid_distances = pairwise_distances(centroids, metric = metric)

    # 如果类内、类间距离非常小，接近于0，返回0
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    score = (intra_dists[:, None] + intra_dists) / centroid_distances
    score[score == np.inf] = np.nan # 将无穷大的值转换为nan
    return np.mean(np.nanmax(score, axis=1))


def davies_bouldin_score_eu_cos(X, deltaX, labels):

    X, labels = check_X_y(X, labels)  # 检查X和labels的尺寸
    deltaX, labels = check_X_y(deltaX, labels)  # 检查X和labels的尺寸
    # deltaX = np.diff(X, n = 1, axis = 1)  # 求X的一阶差分
    le = LabelEncoder()
    labels = le.fit_transform(labels)  # 将labels转换为0-k之间的整数
    n_samples, _ = X.shape  # 样本数
    n_labels = len(le.classes_)  # label数
    check_number_of_labels(n_labels, n_samples)  # 检查label数量是否正确

    intra_dists = np.zeros(n_labels)  # 初始化类内距离
    # 初始化聚类中心
    centroids = np.zeros((n_labels, len(X[0])), dtype = np.float)
    centroids_delta = np.zeros((n_labels, len(deltaX[0])), dtype = np.float)

    eu_X = pairwise_distances(X, metric = 'euclidean')
    eu_max, eu_min = np.max(eu_X), np.min(eu_X)
    cos_X = pairwise_distances(deltaX, metric = 'cosine')
    cos_max, cos_min = np.max(cos_X), np.min(cos_X)
    scaling_ratio = (eu_max - eu_min) / (cos_max - cos_min)

    for k in range(n_labels):
        cluster_k = safe_indexing(X, labels == k)  # 索引样本集中标签为k的样本(即第k类)
        delta_k = safe_indexing(deltaX, labels == k)  # 索引一阶差分样本集中的标签为k的样本
        centroid = cluster_k.mean(axis = 0)  # 求第k类样本的中心点
        centroid_delta = delta_k.mean(axis = 0)    # 求第k类一阶差分样本的中心点
        centroids[k] = centroid
        centroids_delta[k] = centroid_delta
        # 求第k类样本到第k类中心的距离的平均值(以metric度量)
        a = 0.5
        b = 1 - a
        intra_dist_eu = np.average(pairwise_distances(cluster_k, [centroid], metric = 'euclidean'))
        intra_dist_cos = np.average(pairwise_distances(delta_k, [centroid_delta], metric = 'cosine'))
        intra_dists[k] = a * intra_dist_eu + b * intra_dist_cos * scaling_ratio 

    # 求不同类中心的距离(以metric度量)
    centroid_distances_eu = pairwise_distances(centroids, metric = 'euclidean')
    centroid_distances_cos = pairwise_distances(centroids_delta, metric = 'cosine')
    centroid_distances = a * centroid_distances_eu  + b * centroid_distances_cos * scaling_ratio

    # 如果类内、类间距离非常小，接近于0，返回0
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    score = (intra_dists[:, None] + intra_dists) / centroid_distances
    score[score == np.inf] = np.nan  # 将无穷大的值转换为nan
    return np.mean(np.nanmax(score, axis=1))