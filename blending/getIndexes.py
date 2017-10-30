'''
  File name: getIndexes.py
  Author: Krishna Bharathala
  Date created: 10/16/2017
'''

import numpy as np

def getIndexes(mask, targetH, targetW, offsetX, offsetY):
  indexes = np.zeros((targetH, targetW))
  mask_x, mask_y = np.nonzero(mask)

  for i in range(len(mask_x)):
    indexes[mask_x[i]+offsetY][mask_y[i]+offsetX] = i+1

  return indexes