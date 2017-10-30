'''
  File name: getCoefficientMatrix.py
  Author: Krishna Bharathala
  Date created: 10/17/2018
'''

import numpy as np

def getCoefficientMatrix(indexes):
  
  coeffA = np.zeros((int(np.amax(indexes)), int(np.amax(indexes))))
  mask_x, mask_y = np.nonzero(indexes)

  for p in range(len(mask_x)):
    i, j = mask_x[p], mask_y[p]

    coeffA[p][p] = 4

    if i-1 >= 0 and indexes[i-1][j] != 0:
      val1 = int(indexes[i-1][j] - 1)
      coeffA[p][val1] = -1

    if j-1 >= 0 and indexes[i][j-1] != 0:
      val2 = int(indexes[i][j-1] - 1)
      coeffA[p][val2] = -1

    if i+1 < indexes.shape[0] and indexes[i+1][j] != 0:
      val3 = int(indexes[i+1][j] - 1)
      coeffA[p][val3] = -1

    if j+1 < indexes.shape[1] and indexes[i][j+1] != 0:
      val4 = int(indexes[i][j+1] - 1)
      coeffA[p][val4] = -1

  return coeffA
