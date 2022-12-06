import argparse
import os
import CaptureFrame_Process
import numpy as np
from sklearn.model_selection import train_test_split
import Localization
import cv2
import scipy.misc


# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./dataset/trainingsvideo.avi')
    parser.add_argument('--output_path', type=str, default='./Output.csv')
    parser.add_argument('--sample_frequency', type=int, default=2)
    parser.add_argument('--save_files', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_hyper_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast_stretch', type=float, default=0.75)
    parser.add_argument('--gaussian_blur_k', type=tuple, default=(5, 5))
    parser.add_argument('--gaussian_blur_sigma', type=float, default=0.75)
    parser.add_argument('--mask_low', type=object, default=[15, 135, 135])
    parser.add_argument('--mask_high', type=object, default=[40, 255, 255])
    parser.add_argument('--threshold_value', type=int, default=50)
    parser.add_argument('--opening_kernel', type=object, default=np.ones((3, 3)))
    parser.add_argument('--hitmiss_kernel', type=object, default=np.ones((2, 5)))
    args = parser.parse_args()
    return args


# define a get_sizes
def test_cross_validation(hyper_args=None, sizes=[0.1]):
    images = []
    groundTruthBoxes = open("BoundingBoxGroundTruth.csv", "r").read().split('\n')
    labels = []
    cap = cv2.VideoCapture(get_args().file_path)
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
    area_intersection = max(0, min(model_box[1][0], test_box[1][0]) - max(model_box[0][0], test_box[0][0])) * max(0, min(model_box[3][1], test_box[3][1]) - max(model_box[0][1], test_box[0][1]))
    print("Intersection: " + str(area_intersection))

    area_union = area_model_box + area_test_box - area_intersection
    overlap = area_intersection / area_union

    return overlap


def evaluate_bounding_boxes(data, labels, hyper_args, test_size):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42,
                                                        shuffle=False)
    plates_train = []
    plates_test = []
    for img in x_train:
        plates_train.append(Localization.plate_detection(img, hyper_args)[1])
    for img in x_test:
        plates_test.append(Localization.plate_detection(img, hyper_args)[1])

    # converts to the same shape cuz for some god forsaken reason it is not
    plates_train = np.squeeze(np.array(plates_train))
    plates_test = np.squeeze(np.array(plates_test))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    score_train = 0
    score_test = 0
    for i in range(len(plates_train)): score_train += evaluate_single_box(plates_train[i], y_train[i]) / len(plates_train)
    for i in range(len(plates_test)): score_test += evaluate_single_box(plates_test[i], y_test[i]) / len(plates_test)

    print("TrainingSet:" + str(score_train*100.0) + "%")
    print("TestSet:" + str(score_test*100.0) + "%")

    return score_test*100.0


def train_model(data, labels, hyper_args, test_sizes):
    best = 0
    best_hyper_arg = None
    best_size = None
    # for hyper_arg in hyper_args:
    for size in test_sizes:
        res = evaluate_bounding_boxes(data, labels, hyper_args, size)
        if res > best:
            best = res
            # best_hyper_arg = hyper_arg
            best_size = size

    print("Best match: " + str(best) + "%\n hyper_arg = " + str(best_hyper_arg) + "\n size = " + str(size))

    return best_hyper_arg, best_size


# In this file, you need to pass three arguments into CaptureFrame_Process function.
if __name__ == '__main__':
    args = get_args()

    if args.output_path is None:
        output_path = os.getcwd()

    else:
        output_path = args.output_path
    file_path = args.file_path
    sample_frequency = args.sample_frequency
    save_files = args.save_files
    test_cross_validation(get_hyper_args())
    CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path, save_files, get_hyper_args())
