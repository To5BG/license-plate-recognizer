import cv2
import numpy as np
import os

from Localization import sharpKernel

sift = None
references = {}

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


def segment_and_recognize(plate_imgs, hyper_args, quickCheck=False):
    global references, sift
    recognized_plates = []

    if sift is None:
        sift = cv2.SIFT_create(1)  # tuka zashto imash 1 maika mu she iba
        create_database("dataset/SameSizeLetters/")
        create_database("dataset/SameSizeNumbers/")
    for plate_img in plate_imgs:
        recognized_plates.append(recognize_plate(plate_img, hyper_args))
    return recognized_plates


# Go through all files in provided filepath, and images to a in-memory dictionary
# this can be optimized later to contain the sifts directly but it is fine for now
def create_database(path):
    global references, sift

    for f in os.listdir(path):
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
            img = cv2.imread(path + f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :np.max(np.where(img != 0)[1] % img.shape[1]) + 1]
            img = cv2.resize(img, (64, 80))
            references[f.split('.')[0]] = img


def recognize_plate(image, hyper_args):
    # preprocessing steps - sharpen image and improve contour results
    img = image.copy()
    img = cv2.GaussianBlur(img, (hyper_args.gaussian_blur_k, hyper_args.gaussian_blur_k),
                           hyper_args.gaussian_blur_sigma)
    img = cv2.bilateralFilter(img, hyper_args.bifilter_k, hyper_args.bifilter_sigma1, hyper_args.bifilter_sigma2)
    img = cv2.filter2D(img, -1, sharpKernel(hyper_args.sharpen_k, hyper_args.sharpen_sigma))

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresholded image for easier contour computation in same format as ground truth
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sort_contours(contours)[0]  # sorts the contours from left to right

    characterImages = segment_plate(contours, thresh)
    result = ""
    for char in characterImages:
        result += str(recognize_character(char))

    return result


def recognize_character(char):
    global references

    bf = cv2.BFMatcher()
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sift = cv2.SIFT_create()

    k1, d1 = sift.detectAndCompute(char, None)
    allDistances = []
    top_matches = []

    for r in references:
        k2, d2 = sift.detectAndCompute(r, None)
        r = cv2.resize(r, (char.shape[1], char.shape[0]))
        # Sharpen the image using a filter
        r = cv2.filter2D(r, -1, kernel)
        matches = bf.knnMatch(d1, d2, k=2)
        good_matches = []

        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)

        distance = sum(m.distance for m in good_matches) / len(good_matches) if len(
            good_matches) > 0 else 1000000000
        top_matches.append(r)
        allDistances.append(distance)

    for i, char in enumerate(top_matches):
        allDistances[i] += difference_score(char, char)
        allDistances[i] /= 2

    index = np.argmin(allDistances)
    best_image = top_matches[index]

    return references.index(best_image)


def segment_plate(character_contours, image, area_threshold=100):
    characters = []
    for c in character_contours:
        area = cv2.contourArea(c)
        if area > area_threshold:
            x, y, w, h = cv2.boundingRect(c)
            img = image[y:y + h, x:x + w]
            img = img[:, :np.max(np.where(img != 0)[1] % img.shape[1]) + 1]
            characters.append(img)
    return characters


# pasted from stack overflow
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
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def difference_score(test_image, reference_character):
    # xor images
    arr = np.bitwise_xor(test_image, reference_character)
    # return the number of non-zero pixels
    return np.sum(arr)

# def sift_descriptor(img):
# 	global ref_sif
# 	_, desc = ref_sif.detectAndCompute(img, None)
# 	return np.average(desc, axis=0)
