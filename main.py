import argparse
import os
import CaptureFrame_Process
import numpy as np
from sklearn.model_selection import train_test_split
import Localization


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
    CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path, save_files, get_hyper_args())
#
#
# def test_cross_validation():
#     input = [0, 36, 72]
#     output = [[[210, 310],
#                [435, 315],
#                [435, 365],
#                [210, 360]],
#               [[205, 255],
#                [345, 295],
#                [340,325],
#                [192, 285]],
#               [[282, 235],
#                 [410, 240],
#                 [408,266],
#                 [285, 265]]
#               ]

def evaluate_single_box(model_box, test_box):
    area_model_box = (model_box[1][1] - model_box[0][1]) * (model_box[2][0] - model_box[1][0])
    area_test_box = (test_box[1][1] - test_box[0][1]) * (test_box[2][0] - test_box[1][0])

    # consider whether you need to invert the np.min and np.max for the fact that y is inverted
    area_intersection = np.max(0,
                               np.min(model_box[1][1], test_box[1][1]) - np.max(model_box[0][1], test_box[0][1])) * \
                        np.max(0, np.min(model_box[3][0], test_box[3][0]) - np.max(model_box[0][0], test_box[0][0]))
    print("Intersection: " + str(area_intersection))

    area_union = area_model_box + area_test_box - area_intersection
    overlap = area_intersection / area_union

    return overlap

def evaluate_bounding_boxes(data, labels, hyper_args, test_size):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42,
                                                        shuffle=True)

    plates_train = Localization.plate_detection(x_train, hyper_args)[0]
    plates_test = Localization.plate_detection(x_test, hyper_args)[0]

    score_train = np.sum(evaluate_single_box(plates_train, y_train)) / len(plates_train)
    score_test = np.sum(evaluate_single_box(plates_test, y_test)) / len(plates_test)

    print("TrainingSet:" + str(score_train) + "%")
    print("TestSet:" + str(score_test)+ "%")

    return score_test

def train_model(data, labels, hyper_args, test_sizes):
    best = 0
    best_hyper_arg = None
    best_size = None
    for hyper_arg in hyper_args:
        for size in test_sizes:
            res = evaluate_bounding_boxes(data, labels, hyper_arg, size)
            if res > best:
                best = res
                best_hyper_arg = hyper_arg
                best_size = size

    print("Best match: " + str(best) + "%\n hyper_arg = " + str(best_hyper_arg) + "\n size = " + str(size))

    return best_hyper_arg, best_size
