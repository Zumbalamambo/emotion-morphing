'''
  File name: reconstructImg.py
  Author: Krishna Bharathala
  Date created: 10/16/2017
'''

import numpy as np
from scipy.sparse import linalg

from getCoefficientMatrix import getCoefficientMatrix

def reconstructImg(indexes, red, green, blue, targetImg):

  resultImg = np.copy(targetImg)

  coeffA = getCoefficientMatrix(indexes)

  A_red = linalg.spsolve(coeffA, red)
  A_red = np.clip(A_red, 0, 255)
  print("done with red")
  A_green = linalg.spsolve(coeffA, green)
  A_green = np.clip(A_green, 0, 255)
  print("done with green")
  A_blue = linalg.spsolve(coeffA, blue)
  A_blue = np.clip(A_blue, 0, 255)
  print("done with blue")

  mask_x, mask_y = np.nonzero(indexes)
  for p in range(len(mask_x)):
    i, j = mask_x[p], mask_y[p]
    val = int(indexes[i][j]-1)
    resultImg[i, j, :] = np.array([A_red[val], A_green[val], A_blue[val]], dtype=np.uint8)

  return resultImg