import argparse
import os
import CaptureFrame_Process
import numpy as np
from sklearn.model_selection import train_test_split
import Localization
import cv2

# define a get_sizes
def cross_validation(file_path, hyper_args, sizes=[0.1]):
    images = []
    groundTruthBoxes = open("BoundingBoxGroundTruth.csv", "r").read().split('\n')
    labels = []
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened() == False: print("Error opening video stream or file")
    csvLine = 0
    nextFrame = int(groundTruthBoxes[csvLine + 1].split(',')[-2])
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if nextFrame == frame_count:
            csvLine += 1
            labels.append(np.array([[int(a), int(b)] for a, b in zip(groundTruthBoxes[csvLine].split(',')[0:8:2],
                                                                     groundTruthBoxes[csvLine].split(',')[1:8:2])]))
            images.append(frame)
            nextFrame = int(groundTruthBoxes[csvLine + 1].split(',')[-2])
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    train_model(images, labels, hyper_args, sizes)


def evaluate_single_box(model_box, test_box):
    area_model_box = (model_box[1][0] - model_box[0][1]) * (model_box[2][1] - model_box[1][1])
    area_test_box = (test_box[1][0] - test_box[0][0]) * (test_box[2][1] - test_box[1][1])

    # consider whether you need to invert the np.min and np.max for the fact that y is inverted
    area_intersection = max(0, min(model_box[1][0], test_box[1][0]) - max(model_box[0][0], test_box[0][0])) * \
                        max(0,min(model_box[3][1], test_box[3][1]) - max(model_box[0][1], test_box[0][1]))
    #print("Intersection: " + str(area_intersection))

    area_union = area_model_box + area_test_box - area_intersection
    overlap = area_intersection / area_union

    return max(overlap, 0)  # else it is - 700% , idk , too bad :)


def evaluate_bounding_boxes(x_train, x_test, y_train, y_test, hyper_args, size):
    plates_train = []
    plates_test = []
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    default = np.zeros(y_train[0].shape)

    for img in x_train:
        a = Localization.plate_detection(img, hyper_args)[1]
        plates_train.append(a[0] if len(a) > 0 else default)
    for img in x_test:
        a = Localization.plate_detection(img, hyper_args)[1]
        plates_test.append(a[0] if len(a) > 0 else default)

    # converts to the same shape cuz for some god forsaken reason it is not
    plates_train = np.squeeze(np.array(plates_train))
    plates_test = np.squeeze(np.array(plates_test))

    score_train = 0
    score_test = 0
    for i in range(len(plates_train)): score_train += evaluate_single_box(plates_train[i], y_train[i])
    for i in range(len(plates_test)): score_test += evaluate_single_box(plates_test[i], y_test[i])
    score_train /= (len(plates_train))
    score_test /= (len(plates_test))

    print("Hyper parameters:" + str(hyper_args))
    print("Size:" + str(size))
    print("TrainingSet:" + str(score_train * 100.0) + "%")
    print("TestSet:" + str(score_test * 100.0) + "%\n")

    return score_test * 100.0


def train_model(data, labels, hyper_args, sizes):
    best = 0
    best_hyper_arg = None
    best_size = None
    # for hyper_arg in hyper_args:
    for size in sizes:
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=size, random_state=42,
                                                            shuffle=False)
        res = evaluate_bounding_boxes(x_train, x_test, y_train, y_test, hyper_args, size)
        if res > best:
            best = res
            # best_hyper_arg = hyper_arg
            best_size = size

    print("Best match: " + str(best) + "%\n hyper_arg = " + str(best_hyper_arg) + "\n size = " + str(size))

    return best_hyper_arg, best_size
