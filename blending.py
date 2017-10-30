'''
  File name: seamlessCloningPoisson.py
  Author: Krishna Bharathala
  Date created: 10/16/2017
'''

import scipy.sparse
from scipy.sparse import linalg
import numpy as np

def getIndexes(mask, targetH, targetW, offsetX, offsetY):
  # get the indices of non zero pixels in the input mask
  indexes = np.zeros((targetH, targetW))
  [r, c] = np.nonzero(mask)
  # offset the indices into the target image
  target_indices = np.array([r + offsetY, c + offsetX])
  # remove any pixels that do not fit into the image
  query = np.where((target_indices[0] < targetH) & (target_indices[1] < targetW))
  in_bounds = target_indices[:, query].reshape((2, len(query[0])))

  for i in range(1, len(in_bounds[0]) + 1):
    pixel = in_bounds[:, i - 1]
    indexes[pixel[0], pixel[1]] = i 
  return indexes.astype(int)

NUM_NEIGHBORS = 4

def getCoefficientMatrix(indexes):
  num_pixels = np.max(indexes)
  (h, w) = indexes.shape
  coeffA = np.zeros((num_pixels, num_pixels))
  np.fill_diagonal(coeffA, NUM_NEIGHBORS) 

  p_idx = np.nonzero(indexes)
  for p in range(num_pixels):
    (x, y) = (p_idx[0][p], p_idx[1][p])
    # check up
    if x > 0 and indexes[x - 1, y] != 0:
      coeffA[p, indexes[x - 1, y] - 1] = -1
    # check down
    if x < h - 1 and indexes[x + 1, y] != 0:
      coeffA[p, indexes[x + 1, y] - 1] = -1
    # check left
    if y < w - 1 and indexes[x, y + 1] != 0:
      coeffA[p, indexes[x, y + 1] - 1] = -1
    # check right
    if y > 0 and indexes[x, y - 1] != 0:
      coeffA[p, indexes[x, y - 1] - 1] = -1
  return coeffA

NUM_NEIGHBORS = 4

def getSolutionVect(indexes, source, target, offsetX, offsetY):
  N = np.max(indexes)
  (hi, wi) = indexes.shape
  (ht, wt) = source.shape
  SolVectorb = np.zeros(N)
  # relplacement pixel indices in the target image
  p_idx = np.nonzero(indexes)
  for p in range(N):
    (x, y) = (p_idx[0][p], p_idx[1][p])
    (sx, sy) = (x - offsetY, y - offsetX)
    # first we calculate 4xGp
    sol_p = NUM_NEIGHBORS * source[sx, sy]
    # next, for each neighbor...
    if x > 0:  # -------- UP ------------
      # subtract G_n
      if sx > 0:
        sol_p = sol_p - source[max(0, sx - 1), sy]
      if indexes[x - 1, y] == 0:
        # add target f' at this position
        sol_p = sol_p + target[x - 1, y]
    if x < hi - 1:  # ------ DOWN -----------
      # subtract G_n
      if sx < ht:
        sol_p = sol_p - source[min(sx + 1, ht - 1), sy]
      if indexes[x + 1, y] == 0:
        # add target f' at this position
        sol_p = sol_p + target[x + 1, y]
    if y > 0:  # ------- LEFT -----------
      # subtract G_n
      if sy > 0:
        sol_p = sol_p - source[sx, max(sy - 1, 0)]
      if indexes[x, y - 1] == 0:
        # add target f' at this position
        sol_p = sol_p + target[x, y - 1]
    if y < wi - 1:  # ------ RIGHT ----------
      # subtract G_n
      if sx < ht:
        sol_p = sol_p - source[sx, min(sy + 1, wt - 1)]
      if indexes[x, y + 1] == 0:
        # add target f' at this position
        sol_p = sol_p + target[x, y + 1]

    SolVectorb[p] = sol_p

  return SolVectorb

def reconstructImg(indexes, red, green, blue, targetImg):
  resultImg = targetImg.copy()
  N = red.shape[0]

  for p in range(1, N + 1):
    pos = np.where(indexes == p)
    resultImg[pos[0][0], pos[1][0], :] = np.array([red[p-1], green[p-1], blue[p-1]])

  return resultImg

def bound_colors(c):
  c[np.where(c < 0)] = 0
  c[np.where(c > 255)] = 255
  return c

def seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY):
  (targetH, targetW, _) = targetImg.shape
  indexes = getIndexes(mask, targetH, targetW, offsetX, offsetY)
  A = getCoefficientMatrix(indexes)
  N = A.shape[0]
  sol_r = getSolutionVect(indexes, sourceImg[:, :, 0], targetImg[:, :, 0], offsetX, offsetY)
  sol_g = getSolutionVect(indexes, sourceImg[:, :, 1], targetImg[:, :, 1], offsetX, offsetY)
  sol_b = getSolutionVect(indexes, sourceImg[:, :, 2], targetImg[:, :, 2], offsetX, offsetY)
  x_red = scipy.sparse.linalg.spsolve(A, sol_r.reshape(N, 1))
  x_red = bound_colors(x_red)
  x_green = scipy.sparse.linalg.spsolve(A, sol_g.reshape(N, 1))
  x_green = bound_colors(x_green)
  x_blue = scipy.sparse.linalg.spsolve(A, sol_b.reshape(N, 1))
  x_blue = bound_colors(x_blue)
  resultImg = reconstructImg(indexes, x_red, x_green, x_blue, targetImg)
  return resultImg

