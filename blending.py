'''
  File name: seamlessCloningPoisson.py
  Author: Krishna Bharathala
  Date created: 10/16/2017
'''

from scipy.sparse import linalg
import numpy as np

from getIndexes import getIndexes
from getCoefficientMatrix import getCoefficientMatrix
from getSolutionVect import getSolutionVect
from reconstructImg import reconstructImg

def getIndexes(mask, targetH, targetW, offsetX, offsetY):
  indexes = np.zeros((targetH, targetW))
  mask_x, mask_y = np.nonzero(mask)

  for i in range(len(mask_x)):
    indexes[mask_x[i]+offsetY][mask_y[i]+offsetX] = i+1

  return indexes

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

def seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY):

	targetH, targetW = targetImg.shape[0], targetImg.shape[1]
	indexes = getIndexes(mask, targetH, targetW, offsetX, offsetY)

	red = getSolutionVect(indexes, sourceImg[:,:,0], targetImg[:,:,0], offsetX, offsetY)
	green = getSolutionVect(indexes, sourceImg[:,:,1], targetImg[:,:,1], offsetX, offsetY)
	blue = getSolutionVect(indexes, sourceImg[:,:,2], targetImg[:,:,2], offsetX, offsetY)

	resultImg = reconstructImg(indexes, red, green, blue, targetImg)

	return resultImg

