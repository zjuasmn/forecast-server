# ----------------------------------- #
import numpy as np 
# ----------------------------------- #


def performance(y_true, y_pred, quantiles):
  """
  :param y_true:
    true Y, ndarray, shape: (n_sample, 1)
  :param y_pred:
    output of model, ndarray, shape: (n_sample, n_quantile)
  :param quantiles:
    quantiles, ndarray, shape:  (n_quantile,)  
  :return:
    mae, value
    acd, value
    sc, value
  """
  
  y_true = np.round(y_true, decimals = 4)
  y_pred = np.round(y_pred, decimals = 4)

  alphas = []
  n_sample, n_quantile = y_pred.shape
  n_alpha = int((n_quantile - 1) / 2)
  for i in range(n_alpha):
    alpha = round(1 - (quantiles[n_quantile - 1 - i] - quantiles[i]), 2)
    alphas.append(alpha)

  # mae
  mae = np.abs((y_true - y_pred[:, n_alpha][:, np.newaxis])).sum() / n_sample
  
  # acd, skill_score(sc)
  acds = np.zeros((n_alpha, 1))
  scs = np.zeros((n_alpha, 1))
  for i in range(n_alpha):
    lb = y_pred[:, i][:, np.newaxis]  # 取下限值
    ub = y_pred[:, n_quantile - 1 - i][:, np.newaxis]  # 取上限值

    # acd
    in_interval = sum([(lb <= y_true)[j, 0] and (ub >= y_true)[j, 0] for j in range(n_sample)])
    picp = in_interval / n_sample
    acd = picp - (1 - alpha)
    acds[i, 0] = acd

    # skill_score
    interval_width = ub - lb
    l_bias = y_true - lb 
    u_bias = ub - y_true
    sc = (-2 * alpha * interval_width + 4 * l_bias * (y_true <= lb) + 4 * u_bias * (y_true >= ub)).sum() / n_sample
    scs[i, 0] = sc
  

  return mae, acds.mean(), scs.mean()
