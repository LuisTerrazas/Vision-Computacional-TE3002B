import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

for i in range(10):
	img = cv.imread(str(f'Images_sign/img{i}.png'))
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	_,th1 = cv.threshold(gray, 130, 255, cv.THRESH_BINARY)

	cv.imshow(str(f"Original Image: {i}"), img)
	cv.imshow(str(f"Grayscale Image: {i}"), gray)
	cv.imshow(str(f"Binary Image: {i}"),th1)
	cv.waitKey(0)
	cv.destroyAllWindows()

	gaussian_noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
	val_Gauss = 20
	list_Gass = []
	for j in range(3):
		cv.randn(gaussian_noise,128,val_Gauss)
		gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
		noisy_image = cv.add(gray, gaussian_noise)
		list_Gass.append(noisy_image)
		val_Gauss = val_Gauss*3

	Gauss_N = np.concatenate(list_Gass, axis=1)

	cv.imshow("Gaussian Noise", Gauss_N)
	cv.waitKey(0)

	Smooth_GasBlur = []
	Smooth_Bilateral = []
	Smooth_Median = []
	for j in list_Gass:
		Smooth_GasBlur.append(cv.GaussianBlur(j,(5,5), 0))
		Smooth_Median.append(cv.medianBlur(j,5))
		Smooth_Bilateral.append(cv.bilateralFilter(j,9,75,75))


	GasBlur = np.concatenate(Smooth_GasBlur,axis=1)
	Bilateral = np.concatenate(Smooth_Bilateral, axis=1)
	Median = np.concatenate(Smooth_Median, axis=1)

	cv.imshow("Smoothing_GaussianBlur", GasBlur)
	cv.waitKey(0)
	cv.imshow("Smoothin_Bilateral", Bilateral)
	cv.waitKey(0)
	cv.imshow("Smoothing_Median", Median)
	cv.waitKey(0)
	cv.destroyAllWindows()






