import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import Recognize
import Localization


def cross_validate(file_path, hyper_args):
    plates = []
    names = []

    # uses sample recognition dataset
    for f in os.listdir(file_path):
        names.append(f.split(".")[0])
        img = cv2.imread(file_path + "/" + f)
        plates.append(img)

    train_and_test_model_recognition(plates, names, hyper_args)

def train_and_test_model_recognition(x, y, hyper_args):
    best_hyper_arg = None
    hyper_args = np.array([hyper_args])
    test_X = None
    test_Y = None
    best_train = 0
    best_output = None

    for hyper_arg in hyper_args:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)
        matches_plates, matches_chars, output_plates = evaluate_plates(x_train, y_train, hyper_arg)
        if matches_plates > best_train:  # natural selection of results that improve
            best_train = matches_plates
            best_hyper_arg = hyper_arg
            test_X = x_test
            test_Y = y_test

    best_matches_plates, best_matches_chars, best_output = evaluate_plates(test_X, test_Y, best_hyper_arg)

    print("Best Percentage of License Plates:", best_matches_plates)
    print("Best Percentage of Characters:", best_matches_plates)

    print("\nBest match:  ")
    print("Train set: " + str(best_output))
    print("Test set: " + str(test_Y))
    # print("Best hyper-parameters: " + str(best_hyper_arg))
    return best_hyper_arg


def evaluate_plates(images, ground_truth, hyper_args):
    recognized_plates = 0
    percentage = 0

    plates = Recognize.segment_and_recognize(images, hyper_args)

    for i, plate in enumerate(images):
        res = evaluate_single_plate(plate, ground_truth[i], hyper_args)
        recognized_plates += res[0]
        percentage += res[1]

    recognized_plates /= len(images)
    percentage /= len(images)

    print("Percentage of Recognized Plates:", recognized_plates * 100, "%")
    print("Percentage of Recognized Characters:", percentage * 100, "%")

    return recognized_plates, percentage, plates


def evaluate_single_plate(plate, label, hyper_args):
    success = 0  # 0 if even a single char in plate is not found
    numChars = 0  # returns the percentage of characters recognized in the plate

    if plate == label:
        success = 1
    arr = [*label]
    for i, char in enumerate(plate):
        if char == arr[i]:
            numChars += 1

    numChars /= len(arr)

    print(plate, label, success, "\n")

    return success, numChars
