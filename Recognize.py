import cv2
import numpy as np
import os

import Localization

sift = None
reference_images = {}
ref_sift_desc = {}
bf = None

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
def segment_and_recognize(plate_imgs, hyper_args, debug=False, quick_check=False):
    recognized_plates = []
    global sift, bf
    # If not-instantiated, create new SIFT feature extractor and load reference database
    if sift is None:
        sift = cv2.SIFT_create(1)
        create_database("dataset/SameSizeLetters/")
        create_database("dataset/SameSizeNumbers/")
        bf = cv2.BFMatcher.create()
    for i, plate_img in enumerate(plate_imgs):
        recognized_plates.append(recognize_plate(plate_img, i, hyper_args, debug, quick_check))
    return recognized_plates

# Go through all files in provided filepath, and images to a in-memory dictionary
# this can be optimized later to contain the sifts directly but it is fine for now
def create_database(path):
    global reference_images, sift, ref_sift_desc
    for f in os.listdir(path):
        # Look only for relevant image formats
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
            img = cv2.imread(path + f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Provided train letters contain a lot of black spaces, crop that away
            img = img[:, :np.max(np.where(img != 0)[1] % img.shape[1]) + 1]
            # Resize to modulo 16 size for SIFT
            img = cv2.resize(img, (64, 80))
            # Store ref image and sift descriptor in global lookup dictionaries
            reference_images[f.split('.')[0]] = img
            ref_sift_desc[f.split('.')[0]] = sift_descriptor(img)

# Converts the license plate into an image suitable for cutting up and xor-ing and outputs
# the final recognition result
def recognize_plate(image, n, hyper_args, debug, quick_check):
    if quick_check:
        characterImages = segment_plate(image, n, hyper_args, debug)
        if len(characterImages) == 0: return 'F'
        _, d = recognize_character(char, i, debug)
        if d < 2000: res = 'T'
        return 'F'
    # preprocessing steps - sharpen image and improve contour results
    img = image.copy()
    if hyper_args.contrast_stretch != 0:
        img = Localization.contrastImprovementContrastStretching(img, hyper_args.contrast_stretch, 0, 255)
	# Blur to remove noise
    img = cv2.bilateralFilter(img, hyper_args.bifilter_k, hyper_args.bifilter_sigma1, hyper_args.bifilter_sigma2)
	# Sharpen edges
    img = cv2.filter2D(img, -1, Localization.sharpKernel(hyper_args.sharpen_k, hyper_args.sharpen_sigma))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold image
    img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    # Invert threshold if there are more white than black pixels
    if len(np.where(img[5:(len(img) - 5)] == 255)[0]) > len(np.where(img[5:(len(img) - 5)] == 0)[0]):
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]

    # Apply morphological operations
    # Reduce upper and lower borders with hitmiss morph op
    img = cv2.absdiff(img, cv2.morphologyEx(img, cv2.MORPH_HITMISS, hyper_args.hitmiss_kernel))
    img = cv2.absdiff(img, cv2.morphologyEx(img, cv2.MORPH_HITMISS, hyper_args.hitmiss_kernel))
    # Reduce noise with opening morph op, ellipse kernel
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, hyper_args.opening_kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, opening_kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2)), iterations=1)

    # Get white pixels per row/height pixel
    rows = np.array([len(np.where(img[i] == 255)[0]) for i in range(0, image.shape[0])])
    # Potential indices for crop are where the pixels are below a threshold
    potential_idx = np.where(rows <= hyper_args.horizontal_border_low_threshold)[0]
    potential_idx = np.append(potential_idx, 0)
    potential_idx = np.append(potential_idx, img.shape[0])
    img = img[(max(3, np.max(potential_idx[potential_idx <= 12])) - 3)
        :(min(len(img) - 4, np.min(potential_idx[potential_idx >= 38])) + 3)]

    # If debug is enabled, show thresholded, morphed, and filtered image
    if debug:
        cv2.imshow("Preprocessed license plate%d" % n, img)

    # Segment into characters, and accumulate recognitions
    characterImages = segment_plate(img, n, hyper_args, debug)
    res = ""
    if len(characterImages) == 0: return res
    for i, char in enumerate(characterImages):
        res += str(recognize_character(char, i, debug)[0])
    return res

# Using cv2's SIFT implementation directly - approved from Lab_6_Find_Contours_SIFT
def sift_descriptor(img):
    global sift
    _, desc = sift.detectAndCompute(img, None)
    # If no features captured, return empty descriptor
    if desc is None or len(desc) == 0: return []
    # Else average over all keypoints
    return np.average(desc, axis=0)
 
def diff_score_sift(test, ref):
    global bf
    matches = bf.knnMatch(test, ref, k=1)
    return np.sum(list(map(lambda x:x[0].distance, matches)))

def diff_score_xor(test, ref):
    # return the number of non-zero pixels after xoring
    return len(np.where(cv2.bitwise_xor(test, ref) != 0)[0])

def recognize_character(char, n, debug):
    global ref_sift_desc
    cnts, _ = cv2.findContours(char, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0: return ''
    cnt = sorted(cnts, key = cv2.contourArea, reverse=True)[0]

    x, y, w, h = cv2.boundingRect(cnt)
    char = char[y : y + h, x : x + w]
    char = cv2.resize(char, (64, 80))
    if debug:
        cv2.imshow("Character#%d" % n, char)

    #char_sift_desc = sift_descriptor(char)
    #print(char_sift_desc)
    #if len(char_sift_desc) == 0: return ''
    #scores = {k : diff_score_sift(char_sift_desc, ref) for k, ref in ref_sift_desc.items()}
    scores = {k : diff_score_xor(char, ref) for k, ref in reference_images.items()}
    # Check if the ratio of the two scores is close to 1 (if so return empty)
    low1, low2 = sorted(scores.items(), key=lambda x: x[1])[:2]
    return low1
    #if abs(low1[1] / low2[1] - 1) > 0.1: return low1
    #return ''

# Function to segment the plate into individual characters
def segment_plate(image, n, hyper_args, debug):
    # Get white pixels per column/width pixel
    columns = np.array([len(np.where(image[:, i] == 255)[0]) for i in range(0, image.shape[1])])
    # Potential indices for crop are where the pixels are below a threshold
    potential_idx = np.where(columns <= hyper_args.vertical_border_low_threshold)[0]
    # Keep track of current indices, when a new index is encountered that is
    # sufficiently far away from the rest, average the accumulator and add that as an index
    # For the first run take the rightmost entry, similarly for last run take the leftmost entry
    border_idx = []
    curr = []
    if len(potential_idx) == 0: potential_idx = np.insert(potential_idx, 0, 0)
    last = potential_idx[0]
    average = False
    for idx in potential_idx:
        # If jump is sufficiently large
        if idx - last >= hyper_args.min_char_jump:
            # Average the indices for all but first and last cut
            if average: border_idx.append(int(np.mean(curr) + 0.5))
            # Else take rightmost index for first run
            else:
                border_idx.append(max(int(np.max(curr)), 1) - 1)
                average = True
            # Reset curr array for next cut area
            curr = []
        # Append index to current area accumulator
        curr.append(idx)
        # Update last checked index for jump check
        last = idx
    # Add the rightmost image boundary in case of an empty accumulator
    curr.append(image.shape[1] - 2)
    # Last cut on the leftmost possible index
    border_idx.append(int(np.min(curr)) + 1)

    # If debug is enabled, draw all character borders prior to segmentation
    if debug:
        img = image.copy()
        for b in border_idx:
            img = cv2.line(img, (b, 0), (b, 50), (255, 255, 255), 1)
        cv2.imshow("Borders for plate%d" % n, img)

    # For each border idx, cut the image from last border to current border
    images = []
    # If a few borders/segments - return empty arr
    if (len(border_idx) <= 5): return images
    last = border_idx[0]
    for b in range(1, len(border_idx)):
        curr = border_idx[b]
        # Crop image between two borders (last and curr)
        curr_img = image[:, last:(curr + 1)]
        # For cropped image, get white pixels per row to determine if a character is captured
        rows = np.array([len(np.where(curr_img[i] == 255)[0]) for i in range(0, image.shape[0])])
        # If not enough rows have sufficient count of white pixels - consider fluke/dash -> skip
        if len(np.where(rows > hyper_args.horizontal_char_low_threshold)[0]) < hyper_args.char_segment_threshold: continue
        images.append(curr_img)
        last = curr
    return images