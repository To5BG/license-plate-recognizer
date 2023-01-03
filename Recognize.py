import cv2
import numpy as np
import os

from Localization import sharpKernel

sift = None
reference_sift_desc = {}

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""
def segment_and_recognize(plate_imgs, hyper_args, quickCheck = False):
	global reference_sift_desc, sift
	recognized_plates = []

	if sift is None:
		sift = cv2.SIFT_create(1) #tuka zashto imash 1 maika mu she iba
		create_database("dataset/SameSizeLetters/")
		create_database("dataset/SameSizeNumbers/")
	for plate_img in plate_imgs:
		recognized_plates.append(recognize_plate(plate_img, hyper_args))
	return recognized_plates

# Go through all files in provided filepath, and save sift descriptors to a in-memory dictionary
def create_database(path):
	global reference_sift_desc

	for f in os.listdir(path):
		if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
			img = cv2.imread(path + f)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = img[:, :np.max(np.where(img != 0)[1] % img.shape[1]) + 1]
			img = cv2.resize(img, (64, 80))
			reference_sift_desc[f.split('.')[0]] = sift_descriptor(img)

def recognize_plate(image, hyper_args):
	#preprocessing steps
	img = image.copy()
	img = cv2.GaussianBlur(img, (hyper_args.gaussian_blur_k, hyper_args.gaussian_blur_k), hyper_args.gaussian_blur_sigma)
	img = cv2.bilateralFilter(img, hyper_args.bifilter_k, hyper_args.bifilter_sigma1, hyper_args.bifilter_sigma2)
	img = cv2.filter2D(img, -1, sharpKernel(hyper_args.sharpen_k, hyper_args.sharpen_sigma))

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1] #thresholded image for easier contour computation

	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	contours = sort_contours(contours)[0] #sorts the contours from left to right

	characterImages = segment_plate(contours)
	result = ""
	for char in characterImages:
		result += str(recognize_character(char))

	result = "dwadadwa"

	return result

def recognize_character(char):
	global reference_sift_desc

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True) #check documentation for this, see if you can use it
	return ""

def segment_plate(character_contours, image, area_threshold = 10):
	characters = []
	for c in character_contours:
		area = cv2.contourArea(c)
		if area > area_threshold:
			x, y, w, h = cv2.boundingRect(c)
			characters.append(image[y:y + h, x:x + w])
	return characters

#pasted from stack overflow
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


# Using cv2's SIFT implementation directly - approved from Lab_6_Find_Contours_SIFT
def sift_descriptor(img):
	global sift
	keypoints, desc = sift.detectAndCompute(img, None)
	return np.average(desc, axis=0)

# def sift_descriptor(img):
# 	global ref_sif
# 	_, desc = ref_sif.detectAndCompute(img, None)
# 	return np.average(desc, axis=0)
