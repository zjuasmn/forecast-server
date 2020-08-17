# ----------------------------------- #
import numpy as np
import pulp as pl
# ----------------------------------- #

class DQR(object):
  def __init__(self, quantiles, n_feature, n_hidden, n_output):
    """
    :param quantiles:
        quantiles, ndarray, shape:  (n_quantile,)
    :param n_feature:
        number of features
    :param n_hidden:
        number of hidden neurons
    :param n_output:
        number of output
    """
    self.quantiles = quantiles
    self.n_quantile = quantiles.shape[0]
    self.n_feature = n_feature
    self.n_hidden = n_hidden
    self.n_output = n_output
    self.func = 'tanh'  # activation func
    self.random_state = 1  # random seed
    self.b = None
    self.w = None
    self.W = None
    self.x_max = None
    self.x_min = None
    self.y_max = None
    self.y_min = None

  def activate(self, H):
    if self.func == 'tanh':
        return np.tanh(H)
    if self.func == 'sigmoid':
        return 1 / (1 + np.exp(-H))
    if self.func == 'relu':
        return np.maximum(H, 0)
    return H

  """
  def normalize(self, *args):
    if len(args) == 2:
      x, y = args[0], args[1]
      self.x_max, self.x_min = np.max(x, axis=0), np.min(x, axis=0)
      self.y_max, self.y_min = np.max(y, axis=0), np.min(y, axis=0)
      x = (x - self.x_min) / (self.x_max - self.x_min)
      y = (y - self.y_min) / (self.y_max - self.y_min)
      return x, y
    if len(args) == 1:
      x = args[0]
      x = (x - self.x_min) / (self.x_max - self.x_min)
      return x

  def de_normalize(self, y):
    return y * (self.y_max - self.y_min) + self.y_min
  """

  def fit(self, x, y):
    """
    :param x:
        train input, ndarray, shape: (n_sample, n_feature)
    :param y:
        train output, ndarray, shape: (n_sample, 1)
    :return:
        None
    """
    # number of samples
    n_sample = x.shape[0]

    # normalization
    # x, y = self.normalize(x, y)

    # get ELM parameters
    np.random.seed(self.random_state)
    array = np.random.normal(size=[self.n_feature + 1, self.n_hidden])
    self.w, self.b = array[:-1, :], array[-1, :]
    bb = np.repeat(np.expand_dims(self.b, axis=0), n_sample, axis=0)
    H = np.dot(x, self.w) + bb
    # self.func = Layer['func'].lower()
    H = self.activate(H)

    # make LP
    prob = pl.LpProblem("myProblem", pl.LpMinimize)

    # set variables --- f_alpha_time, w_alpha_hidden
    # i think set the bound for weight is necessary,
    # otherwise it would be likely to get thousand and more
    # which result in unstable prediction
    ff = pl.LpVariable.dicts('f', indexs=(range(self.n_quantile), range(n_sample)))
    ww = pl.LpVariable.dicts('w', indexs=(range(self.n_quantile), range(self.n_hidden)))
    # ww = pl.LpVariable.dicts('w', indexs=(range(self.n_quantile), range(self.n_hidden)), lowBound=-10, upBound=10)

    # objection
    prob += pl.lpSum([1.0 * ff[i][j] for i in range(self.n_quantile) for j in range(n_sample)])

    # constrains
    for i in range(self.n_quantile):
      qt0, qt1 = 1 / self.quantiles[i], 1 / (self.quantiles[i] - 1)
      for j in range(n_sample):
        # Hw
        # 这里为什么是个列表，不应该求和吗？
        # 相当于所有分位点共用一套输入权重与偏置？
        temp1 = [H[j, k] * ww[i][k] for k in range(self.n_hidden)]  
        prob += pl.lpSum([qt0 * ff[i][j]] + temp1 + [-y[j]]) >= 0
        prob += pl.lpSum([qt1 * ff[i][j]] + temp1 + [-y[j]]) <= 0  # 是小于0吗？不应该也是大于0？
        prob += pl.lpSum(temp1) >= 0
        prob += pl.lpSum(temp1) <= 1
        if i != 0:
          temp2 = [-H[j, k] * ww[i-1][k] for k in range(self.n_hidden)]
          prob += pl.lpSum(temp1 + temp2) >= 0

    # prob.writeLP("MM.lp")
    # print(prob)

    # solve the LP
    # pl.list_solvers(1)
    # 'CPLEX_PY', 'PULP_CBC_CMD'(default), 'PULP_CHOCO_CMD'
    solver = pl.get_solver('PULP_CBC_CMD')
    prob.solve(solver)
    # prob.solve()

    # get Weight
    self.W = np.zeros((self.n_hidden, self.n_quantile))
    for i in range(self.n_quantile):
      for j in range(self.n_hidden):
        self.W[j, i] = pl.value(ww[i][j])
    print("objective=", pl.value(prob.objective))

  # 需要将predict函数分拆出去，改成从json文件中读取w，b，W的方式
  def predict(self, x, **kwgs):
    """
    :param x:
      test input, ndarray, shape: (time, feature)
    :param quantiles:
      quantiles, ndarray, shape:  (n_quantile,)
    :param w:
      in weight, ndarray, shape: (n_feature, n_hidden)
    :param b:
      in bias, ndarray, shape: (1, n_hidden)
    :param W:
      out weight, ndarray, shape: (n_hidden, n_quantile)    
    :return:
      test output, ndarray, shape: (time, n_quantile)
    """

    if len(kwgs) == 0:
      quantiles, w, b, W = self.quantiles, self.w, self.b, self.W
    elif len(kwgs) == 4:
      quantiles, w, b, W = kwgs['quantiles'], kwgs['w'], kwgs['b'], kwgs['W']
    
    # x = self.normalize(x)
    n_sample = x.shape[0]
    n_quantile = quantiles.shape[0]

    # cal ELM result
    bb = np.repeat(np.expand_dims(b, axis=0), n_sample, axis=0)  # shape: (n_sample, n_hidden)
    H = np.dot(x, w) + bb
    H = self.activate(H)

    # predict
    y = np.zeros((n_sample, n_quantile))  # shape: (n_sample, n_quantile)
    for i in range(n_quantile):
      y[:, i] = np.dot(H, W[:, i])

    return np.minimum(np.maximum(y, 0), 1)



