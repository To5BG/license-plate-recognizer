import cv2
import numpy as np
import os
import re

import Localization

reference_images = {}

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
    if len(reference_images) == 0:
        create_database("dataset/SameSizeLetters/")
        create_database("dataset/SameSizeNumbers/")
    for i, plate_img in enumerate(plate_imgs):
        recognized_plates.append(recognize_plate(plate_img, i, hyper_args, debug, quick_check))
    return recognized_plates

# Go through all files in provided filepath, and images to a in-memory dictionary
def create_database(path):
    global reference_images
    for f in os.listdir(path):
        # Look only for relevant image formats
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
            img = cv2.imread(path + f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Provided train letters contain a lot of black spaces, crop that away
            img = img[:, :np.max(np.where(img != 0)[1] % img.shape[1]) + 1]
            img = cv2.threshold(cv2.resize(img, (64, 80)), 128, 255, cv2.THRESH_BINARY)[1]
            # Store ref image in global lookup dictionaries
            letter = re.split("_|\.", f)[0]
            if letter not in reference_images.keys():
                reference_images[letter] = []
            reference_images[letter].append(img)

# Converts the license plate into an image suitable for cutting up and xor-ing and outputs
# the final recognition result
def recognize_plate(image, n, hyper_args, debug, quick_check):
    # preprocessing steps - sharpen image and improve contour results
    img = image.copy()
    if hyper_args.contrast_stretch != 0:
        img = Localization.contrastImprovementContrastStretching(img, hyper_args.contrast_stretch, 0, 255)
    for i in range(hyper_args.sharpen_iter):
        # Blur to remove noise
        img = cv2.bilateralFilter(img, hyper_args.bifilter_k, hyper_args.bifilter_sigma1, hyper_args.bifilter_sigma2)
	    # Sharpen edges
        img = cv2.filter2D(img, -1, Localization.sharpKernel(hyper_args.sharpen_k, hyper_args.sharpen_sigma))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold image
    img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    # Invert threshold if there are more white than black pixels
    if len(np.where(img[10:(len(img) - 10)] == 255)[0]) > len(np.where(img[10:(len(img) - 10)] == 0)[0]):
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]

    # Apply morphological operations
    # Reduce upper and lower borders with hitmiss morph op
    img = cv2.absdiff(img, cv2.morphologyEx(img, cv2.MORPH_HITMISS, np.ones(hyper_args.hitmiss_kernel_1)))
    img = cv2.absdiff(img, cv2.morphologyEx(img, cv2.MORPH_HITMISS, np.ones(hyper_args.hitmiss_kernel_2)))
    img = cv2.absdiff(img, cv2.morphologyEx(img, cv2.MORPH_HITMISS, np.ones(hyper_args.hitmiss_kernel_3)))
    img = cv2.absdiff(img, cv2.morphologyEx(img, cv2.MORPH_HITMISS, np.ones(hyper_args.hitmiss_kernel_4)))
    # Reduce noise with opening morph op, ellipse kernel
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, hyper_args.opening_kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, opening_kernel, iterations=1)
    # Reduce more noise with other openings
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 1)), iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((1, 2)), iterations=1)

    # Get white pixels per row/height pixel
    rows = np.array([len(np.where(img[i] == 255)[0]) for i in range(0, image.shape[0])])
    # Potential indices for crop are where the pixels are below a threshold
    potential_idx = np.where(rows <= hyper_args.horizontal_border_low_threshold)[0]
    potential_idx = np.append(potential_idx, 0)
    potential_idx = np.append(potential_idx, img.shape[0])
    img = img[np.max(potential_idx[potential_idx <= 12]):np.min(potential_idx[potential_idx >= 38])]

    # If debug is enabled, show thresholded, morphed, and filtered image
    if debug:
        cv2.imshow("Preprocessed license plate%d" % n, img)

    # Segment into characters, and accumulate recognitions
    characterImages = segment_plate(img, n, hyper_args, debug)
    res = ""
    if len(characterImages) == 0: return 'F' if quick_check else res
    tdist = 0
    for i, char_img in enumerate(characterImages):
        char, dist = recognize_character(char_img, i, hyper_args, debug, quick_check)
        if char == '-' and (res == "" or res.endswith('-') or i + 1 == len(characterImages)): continue
        tdist += dist
        res += str(char)
        
    if quick_check: return 'T' if len(res.replace('-','')) > 4 and tdist < hyper_args.plate_dist_threshold else 'F'
    else: return res if len(res) > 4 and tdist < hyper_args.plate_dist_threshold else ""

def diff_score_xor(test, ref):
    res = 0
    for r in ref:
        # return the number of non-zero pixels after xoring
        res += len(np.where(cv2.bitwise_xor(test, r) != 0)[0])
    return res / len(ref)

def recognize_character(char, n, hyper_args, debug, quick_check):
    cnts, _ = cv2.findContours(char, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0: return ''
    cnt = sorted(cnts, key = cv2.contourArea, reverse=True)[0]

    x, y, w, h = cv2.boundingRect(cnt)
    char = char[y : y + h, x : x + w]
    char = cv2.resize(char, (64, 80))
    if debug:
        cv2.imshow("Character#%d" % n, char)
    if len(np.where(char != 0)[0]) / (char.shape[0] * char.shape[1]) > hyper_args.dash_threshold: return ('-', 1000)

    scores = {k : diff_score_xor(char, ref) for k, ref in reference_images.items()}
    # Check if the ratio of the two scores is close to 1 (if so return empty)
    if quick_check:
        l = sorted(scores.items(), key=lambda x: x[1])[0]
        if l[1] < hyper_args.char_dist_threshold: return l
        return ('', 9999)
    else:
        low1, low2 = sorted(scores.items(), key=lambda x: x[1])[:2]
        if (set([low1[0], low2[0]]) in [set(["8", "B"]), set(["0", "D"]), set(["5", "S"]), set(["2", "Z"])]
            and low2[1] / low1[1] < 1.1):
            pass
            #print(low1, low2)
            #return line_check(char, (low1[0], low2[0]))
        if low1[1] < hyper_args.char_dist_threshold: return low1
        return ('', 9999)

# DO EXTRA CHECKS FOR CLOSE PAIRS ((8, B), (0, D), (5, S), and (2, Z))
def line_check(char, chars):
    return chars[0]

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
        # For cropped image, get white pixels per row
        rows = np.array([len(np.where(curr_img[i] == 255)[0]) for i in range(0, image.shape[0])])
        # Threshold rows to determine if a character is captured
        thresholded_rows = np.where(rows < hyper_args.horizontal_char_low_threshold)[0]
        if (
            # If not enough rows have sufficient count of white pixels - consider fluke -> skip
            len(thresholded_rows) > hyper_args.char_segment_threshold and not
            # Unless it is a dash
            len(set(range(image.shape[0])).difference(set(thresholded_rows)).intersection(range(image.shape[0] // 2 - 5, image.shape[0] // 2 + 5))) > 3
            ): continue
        images.append(curr_img)
        last = curr
    return images