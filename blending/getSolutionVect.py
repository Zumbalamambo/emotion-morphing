'''
  File name: getSolutionVect.py
  Author: Krishna Bharathala
  Date created: 10/16/2017
'''

import numpy as np

def getSolutionVect(indexes, source, target, offsetX, offsetY):

  srcH, srcW = source.shape[0], source.shape[1]
  tgtH, tgtW = target.shape[0], target.shape[1]

  SolVectorb = np.zeros(int(np.amax(indexes)))

  mask_x, mask_y = np.nonzero(indexes)

  for i in range(len(mask_x)):
    pi, pj = mask_x[i], mask_y[i]
    src_i, src_j = pi - offsetY, pj - offsetX

    src_color = 4 * source[src_i][src_j]

    if src_i + 1 < srcH:
      src_color -= source[src_i+1][src_j]
    if src_j + 1 < srcW:
      src_color -= source[src_i][src_j+1]
    if src_i - 1 >= 0:
      src_color -= source[src_i-1][src_j]
    if src_j - 1 >= 0:
      src_color -= source[src_i][src_j-1]

    tgt_color = 0
    if pi + 1 < tgtH and indexes[pi+1][pj] == 0:
      tgt_color += target[pi+1][pj]
    if pj + 1 < tgtW and indexes[pi][pj+1] == 0:
      tgt_color += target[pi][pj+1]
    if pi - 1 >= 0 and indexes[pi-1][pj] == 0:
      tgt_color += target[pi-1][pj]
    if pj - 1 >= 0 and indexes[pi][pj-1] == 0:
      tgt_color += target[pi+1][pj-1]    

    SolVectorb[int(indexes[pi][pj]-1)] = src_color + tgt_color

  return SolVectorb
