import cv2
import numpy as np
import os

import Localization

sift = None
reference_images = []
reference_sifts = []
letters = []

"""
In this file, you will define your own segment_and_recognize function.
To do:
    1. Segment the plates character by character
    2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' 
    and 'SameSizeNumbers')
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


def segment_and_recognize(plate_imgs, hyper_args, quick_check=False):
    global reference_images, sift
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
    global reference_images, reference_sifts, sift, letters

    for f in os.listdir(path):
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
            img = cv2.imread(path + f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :np.max(np.where(img != 0)[1] % img.shape[1]) + 1]
            img = cv2.resize(img, (64, 80))
            reference_images.append(img)
            reference_sifts.append(sift.detectAndCompute(img, None))
            letters.append(f.split('.')[0])


# Converts the license plate into an image suitable for cutting up and xor-ing and outputs
# the final recognition result
def recognize_plate(image, hyper_args):
    # preprocessing steps - sharpen image and improve contour results
    img = image.copy()
    img = cv2.GaussianBlur(img, (hyper_args.gaussian_blur_k, hyper_args.gaussian_blur_k),
                           hyper_args.gaussian_blur_sigma)
    img = cv2.bilateralFilter(img, hyper_args.bifilter_k, hyper_args.bifilter_sigma1, hyper_args.bifilter_sigma2)
    img = cv2.filter2D(img, -1, Localization.sharpKernel(hyper_args.sharpen_k, hyper_args.sharpen_sigma))

    # threshold image for easier contour computation in same format as ground truth - will also later be used for
    # xor verification
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sort_contours(contours)[0]  # sorts the contours from left to right

    characterImages = segment_plate(contours, thresh)
    result = ""
    for char in characterImages:
        result += str(recognize_character(char))

    return result


def recognize_character(char):
    global reference_images, sift

    bf = cv2.BFMatcher()
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sift = cv2.SIFT_create()

    k1, d1 = sift.detectAndCompute(char, None)
    allDistances = []
    difference_scores = []
    images = []

    for i, r in enumerate(reference_images):
        k2, d2 = sift.detectAndCompute(r, None) #for some reason it gives a glich with the sift in-mem database
        r = cv2.resize(r, (char.shape[1], char.shape[0]))
        # Sharpen the image using a filter
        r = cv2.filter2D(r, -1, kernel)
        matches = bf.knnMatch(d1, d2, k=2)
        good_matches = []

        # ratio test for sift
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        distance = sum(m.distance for m in good_matches) / len(good_matches) if len(
            good_matches) > 0 else 100000000  # some arbitrarily large value if there are no sift matches at all
        difference_scores.append(difference_score(char, r))
        allDistances.append(distance)
        images.append(r)

    # Normalize both lists
    minV = min(allDistances)
    maxV = max(allDistances)
    allDistances = np.array(allDistances)
    #allDistances = (allDistances - minV) / (maxV - minV)
    difference_scores = np.array(difference_scores)
    minV = min(difference_scores)
    maxV = max(difference_scores)
    difference_scores = (difference_scores - minV) / (maxV - minV)


    for i in range(len(allDistances)):
        #print(allDistances[i])
       # print(difference_scores[i])
        allDistances[i] *= difference_scores[i]
        #allDistances[i] /= 2

    index = np.argmin(allDistances)

    return letters[index]


# function to cut up the plate into individual characters - tests by area size
# and aspect ratio so as not to let through contours within characters
def segment_plate(character_contours, image):
    characters = []
    imageArea = image.shape[0] * image.shape[1]
    for c in character_contours:
        x, y, w, h = cv2.boundingRect(c)

        if np.isclose(h / w, 2, 0.75) and w * h > 0.025 * imageArea:  # testing by area size and aspect ratio
            img = image[y:y + h, x:x + w]
            img = img[:, :np.max(np.where(img != 0)[1] % img.shape[1]) + 1]
            characters.append(img)
    return characters


# attempt to segment based on the black spaces - so far unsuccessful

# def segment_plate(image):
#     edges = cv2.Canny(image,100,200)
#     vertical_projection = np.sum(edges, axis=0)
#     vertical_projection = vertical_projection != 0

#     changes = np.logical_xor(vertical_projection[1:], vertical_projection[:-1])
#     change_pts = np.nonzero(changes)[0]

#     # Extract the individual character images by cropping the original image
#     char_images = []
#     for i in range(len(change_pts)-1):
#         x1, y1 = 0, change_pts[i]
#         x2, y2 = thresh.shape[1], change_pts[i+1]
#         char_image = image[y1:y2, x1:x2]
#         char_images.append(char_image)
#     return char_images


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
    return cnts, boundingBoxes


# xor function used as an additional verification in the recognition algorithm
def difference_score(test_image, reference_character):
    # xor images
    arr = np.bitwise_xor(test_image, reference_character)
    # return the number of non-zero pixels
    return np.sum(arr)
