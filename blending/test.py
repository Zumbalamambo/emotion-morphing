from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from drawMask import draw_mask
from seamlessCloningPoisson import seamlessCloningPoisson
from getCoefficientMatrix import getCoefficientMatrix

def main():
	sourceImg = np.array(Image.open('SourceImage.jpg').convert('RGB'))
	targetImg = np.array(Image.open('TargetImage.jpg').convert('RGB'))

	mask, bbox = draw_mask(sourceImg);

	offsetX, offsetY = 250, 180

	resultImg = seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY)

	plt.imshow(resultImg)
	plt.show()

if __name__ == "__main__":
  main()