import cv2
import numpy as np

last_image = None
last_boxes = list()

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
	img = image.copy()

	if hyper_args.contrast_stretch != 0:
		img = contrastImprovementContrastStretching(img, hyper_args.contrast_stretch, 0, 255)

	# Blur to remove noise
	#img = cv2.bilateralFilter(img, 11, 17, 17)
	img = cv2.GaussianBlur(img, (hyper_args.gaussian_blur_k, hyper_args.gaussian_blur_k), hyper_args.gaussian_blur_sigma)
	# Sharpen edges
	img = cv2.filter2D(img, -1, sharpKernel(hyper_args.sharpen_k, hyper_args.sharpen_sigma))

	# Color segmentation
	# Define color range
	# Similar to lab_1_color_and_histograms color segmentation
	colorMin = np.array(hyper_args.mask_low)
	colorMax = np.array(hyper_args.mask_high)
	# Segment only the selected color from the image and leave out all the rest (apply a mask)
	mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), colorMin, colorMax)
	plate_imgs = cv2.bitwise_and(img, img, mask=mask)

	## Canny's algorithm for edge detection
	edged = cv2.Canny(plate_imgs, hyper_args.canny_lower, hyper_args.canny_upper)

	## Noise reduction after finding segmentation
	# Using morphological filtering
	edged_horizontal = cv2.morphologyEx(edged, cv2.MORPH_OPEN, hyper_args.opening_kernel, iterations=1)
	edged_vertical = cv2.morphologyEx(edged, cv2.MORPH_OPEN, hyper_args.opening_kernel.T, iterations=1)
	edged = cv2.bitwise_or(edged_horizontal, edged_vertical)

	edged_horizontal = cv2.morphologyEx(edged, cv2.MORPH_DILATE, np.ones((1, 2)), iterations=4)
	edged_vertical = cv2.morphologyEx(edged, cv2.MORPH_DILATE, np.ones((2, 1)), iterations=4)
	edged = cv2.bitwise_or(edged_horizontal, edged_vertical)

	#edged_horizontal = cv2.morphologyEx(edged, cv2.MORPH_HITMISS, hyper_args.hitmiss_kernel)
	#edged_vertical = cv2.morphologyEx(edged, cv2.MORPH_HITMISS, hyper_args.hitmiss_kernel.T)
	#edged = cv2.bitwise_or(edged_horizontal, edged_vertical)
	cv2.imshow("Edges detected", edged)
	
	# Find bounding boxes
	boxes = list()
	centers = list()
	plate_imgs = list()
	# By using contour detection
	cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# Save only 3 largest contours
	cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:2]
	for cnt in cnts:
		approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
		rect = cv2.minAreaRect(cnt)
		if len(approx) == 4:
		#if len(approx) == 4 and (len(centers) == 0
		#	or next(filter(lambda centers: abs(centers[0] - rect[0][0]) < 100 and abs(centers[1] - rect[0][1]) < 100, centers), None) is not None):
			box = cv2.boxPoints(rect)
			boxes.append(np.array([box[1], box[2], box[3], box[0]]).astype(np.int32))
			#img = cv2.warpAffine(img, cv2.getRotationMatrix2D(rect[0], rect[2], 1), tuple(map(int, rect[1])))
			#plate_imgs.append(imutils.resize(img, width=hyper_args.image_width))

	#idxy = np.where(threshold != 0)[0]; idxx = np.where(threshold != 0)[1]
	#minx = np.argmin(idxx); miny = np.argmin(idxy); maxx = np.argmax(idxx); maxy = np.argmax(idxy)
	#points = list() + [[idxy[minx], idxx[minx]], [idxy[miny], idxx[miny]], [idxy[maxx], idxx[maxx]], [idxy[maxy], idxx[maxy]]]
	#points.sort(key=lambda p: (p[0] + p[1], p[0]))
	#boxes.append(np.array(points))
	
	# Default position if not able to find a bounding box on current frame
	global last_boxes
	global last_image
	if last_image is None: last_image = image
	if cv2.matchTemplate(image, last_image, 1) > 0.2:
		last_image = image
		last_boxes = list()
	if len(boxes) == 0:
		boxes = last_boxes
	else:
		last_boxes = boxes
		last_image = image
	boxes = np.array(boxes)
	return plate_imgs, boxes

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

# Approved from lab_3_filtering
def sharpKernel(kernel, sigma):
	k = (kernel - 1) // 2
	sharpen_kernel = None
	# Do box filtered sharpen kernel instead
	if sigma < 0.0 or sigma > kernel:
		sharpen_kernel = np.ones((kernel, kernel)) / - kernel ** 2
		sharpen_kernel[k, k] = 2 - 1 / kernel ** 2
	else:
		sharpen_kernel = np.zeros((kernel, kernel))
		sharpen_kernel[k, k] = 2
		g = cv2.getGaussianKernel(kernel, sigma)
		sharpen_kernel -= g @ g.T
	return sharpen_kernel