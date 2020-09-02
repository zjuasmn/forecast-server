import json
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import davies_bouldin_score as dbi
from patternAnalysis import patternAnalysis

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print('usage: cluster.py [clusterInput.json] [clusterOutput.json]')
    exit(1)
  with open(sys.argv[1], encoding='utf-8') as clusterInputFile:
    clusterInput = json.load(clusterInputFile)
  month = clusterInput['month']
  data = clusterInput['data']
  clusterOutput = patternAnalysis(month, data)
  with open(sys.argv[2], encoding='utf_8', mode = 'w') as clusterOutputFile:
    json.dump(clusterOutput, clusterOutputFile, indent=2)
