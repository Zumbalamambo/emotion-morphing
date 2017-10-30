'''
  File name: seamlessCloningPoisson.py
  Author: Krishna Bharathala
  Date created: 10/16/2017
'''

from getIndexes import getIndexes
from getCoefficientMatrix import getCoefficientMatrix
from getSolutionVect import getSolutionVect
from reconstructImg import reconstructImg

def seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY):

	targetH, targetW = targetImg.shape[0], targetImg.shape[1]
	indexes = getIndexes(mask, targetH, targetW, offsetX, offsetY)

	red = getSolutionVect(indexes, sourceImg[:,:,0], targetImg[:,:,0], offsetX, offsetY)
	green = getSolutionVect(indexes, sourceImg[:,:,1], targetImg[:,:,1], offsetX, offsetY)
	blue = getSolutionVect(indexes, sourceImg[:,:,2], targetImg[:,:,2], offsetX, offsetY)

	resultImg = reconstructImg(indexes, red, green, blue, targetImg)

	return resultImg