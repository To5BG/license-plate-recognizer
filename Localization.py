import cv2
import numpy as np

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""
def plate_detection(image, hyper_args):
    # Contrast stretch the image
	if hyper_args.contrast_stretch != 0:
		image = contrastImprovementContrastStretching(image, hyper_args.contrast_stretch, 0, 255)
	# Blur to remove noise
	# Faster implementation to lab_3_filtering implementation of gaussian filter
	image = cv2.GaussianBlur(image, hyper_args.gaussian_blur_k, hyper_args.gaussian_blur_sigma)

	# Color segmentation
	# Define color range
	# Similar to lab_1_color_and_histograms color segmentation
	colorMin = np.array(hyper_args.mask_low)
	colorMax = np.array(hyper_args.mask_high)
	# Segment only the selected color from the image and leave out all the rest (apply a mask)
	mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), colorMin, colorMax)
	plate_imgs = cv2.bitwise_and(image, image, mask=mask)

	## Add Canny's algorithm for edge detection later

	## Noise reduction after finding segmentation
	# Using gaussian blur, thresholding, and morphological filtering
	plate_imgs = cv2.GaussianBlur(plate_imgs, hyper_args.gaussian_blur_k, hyper_args.gaussian_blur_sigma)
	ret, threshold = cv2.threshold(cv2.cvtColor(plate_imgs, cv2.COLOR_BGR2GRAY), hyper_args.threshold_value, 255, cv2.THRESH_BINARY)
	#threshold = cv2.morphologyEx(threshold, cv2.MORPH_DILATE, np.ones((4, 4)))
	#threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, hyper_args.opening_kernel)
	#threshold = cv2.morphologyEx(threshold, cv2.MORPH_HITMISS, hyper_args.hitmiss_kernel)

	boxes = list()
	idxx = np.where(threshold != 0)[0]
	idxy = np.where(threshold != 0)[1]
	#threshold[np.sort( - np.median(threshold))]
	boxes = np.array(boxes, np.int32)
	return threshold, boxes
	return #image[y : y + h, x : x + w], iou

# Approved from lab_1_color_and_histograms
def contrastImprovementContrastStretching(img, contrastFactor, minV, maxV):
	# Guard clause
	q = max(min(contrastFactor, 1), 0)    
	for c in range(img.shape[2]):
		# Calculate global lows and highs (quantiles for percentile stretch, returns min and max (regular stretch) if f = 1.0)
		low = np.quantile(img[:, :, c], (1 - q) / 2)
		high = np.quantile(img[:, :, c], 1 - (1 - q) / 2)
		img[:, :, c] = np.clip((img[:, :, c].astype(np.float16) - low) * (maxV - minV) / (high - low) + minV, 0.0, 255.0).astype(int)
	return img