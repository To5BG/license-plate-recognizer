import cv2
import numpy as np
import os

ref_sif = None
ref_sif_desc = {}

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
	global ref_sif_desc, ref_sif
	recognized_plates = ['dawdawdasd']
	if ref_sif is None:
		ref_sif = cv2.SIFT_create(1)
		create_database("dataset/SameSizeLetters/")
		create_database("dataset/SameSizeNumbers/")
	for plate_img in plate_imgs:
		recognized_plates.append(['dadwadaw'])
	return recognized_plates

# Go through all files in provided filepath, and save sift descriptors to a in-memory dictionary
def create_database(path):
	global ref_sif_desc
	for f in os.listdir(path):
		if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
			img = cv2.imread(path + f)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = img[:, :np.max(np.where(img != 0)[1] % img.shape[1]) + 1]
			img = cv2.resize(img, (64, 80))
			ref_sif_desc[f.split('.')[0]] = sift_descriptor(img)

# Using cv2's SIFT implementation directly - approved from Lab_6_Find_Contours_SIFT
def sift_descriptor(img):
	global ref_sif
	_, desc = ref_sif.detectAndCompute(img, None)
	return np.average(desc, axis=0)
