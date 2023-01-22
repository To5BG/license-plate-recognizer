import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import Recognize
import re
import argparse
from itertools import product


def cross_validate(file_path, hyper_args):
    plates = []
    names = []

    # uses sample recognition dataset
    for f in os.listdir(file_path):
        names.append(re.split("_|\.", f)[0])
        img = cv2.imread(file_path + "/" + f)
        plates.append(img)

    train_and_test_model_recognition(plates, names, hyper_args)

def train_and_test_model_recognition(x, y, hyper_args):
    best_hyper_arg = None
    test_X = None
    test_Y = None
    best_train = 0
    best_output = None

    runs = 0
    for v in product(*hyper_args.values()):
        print(runs)
        runs += 1
        hyper_arg_dict = dict(zip(hyper_args, v))
        parser = argparse.ArgumentParser()
        for k, v in hyper_arg_dict.items():
            parser.add_argument('--' + str(k), type=type(v), default=v)
        hyper_arg = parser.parse_args()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
        matches_plates, _, _ = evaluate_plates(x_train, y_train, hyper_arg)
        if matches_plates > best_train:  # natural selection of results that improve
            best_train = matches_plates
            best_hyper_arg = hyper_arg
            test_X = x_test
            test_Y = y_test

    best_matches_plates, best_matches_chars, best_output = evaluate_plates(test_X, test_Y, best_hyper_arg)

    print("Best Percentage of License Plates:", best_matches_plates)
    print("Best Percentage of Characters:", best_matches_chars)
    #print("\nBest match:  ")
    #print("Train set: " + str(best_output))
    #print("Test set: " + str(test_Y))
    print("Best hyper-parameters: " + str(best_hyper_arg))
    return best_hyper_arg


def evaluate_plates(images, ground_truth, hyper_args):
    recognized_plates = 0
    percentage = 0

    plates = Recognize.segment_and_recognize(images, hyper_args)

    for i, plate in enumerate(plates):
        res = evaluate_single_plate(plate, ground_truth[i])
        recognized_plates += res[0]
        percentage += res[1]

    recognized_plates /= len(images)
    percentage /= len(images)

    #print("Percentage of Recognized Plates:", recognized_plates * 100, "%")
    #print("Percentage of Recognized Characters:", percentage * 100, "%")

    return recognized_plates, percentage, plates

# Evalautes a single plate by comparing plate with label
def evaluate_single_plate(plate, label):
    success = 0  # 0 if even a single char in plate is not found
    numChars = 0  # returns the percentage of characters recognized in the plate

    dist = string_dist(plate, label)
    if dist == 0: success = 1

    maxd = max(len(plate), len(label))
    numChars = (maxd - dist) / maxd
    
    if success == 0: print(plate, label, "\n")
    return success, numChars

# Use levenshtein distance between two strings to determine plate character accuracy
def string_dist(str1, str2):
    # Initialize distance matrix
    m = len(str1)
    n = len(str2)
    dists = np.zeros((m + 1, n + 1))
    for i in range(0, m): dists[i + 1, 0] = i + 1
    for j in range(0, n): dists[0, j + 1] = j + 1
    # Iterate through character pairs
    for j in range(0, n):
        for i in range(0, m):
            if str1[i] == str2[j]: sub = 0
            else: sub = 1
            dists[i + 1, j + 1] = min(dists[i, j + 1] + 1, dists[i + 1, j] + 1, dists[i, j] + sub)
    return dists[m, n]
