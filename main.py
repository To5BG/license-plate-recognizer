import argparse
import os
import CaptureFrame_Process
import numpy as np
import cross_validation


# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./dataset/trainingsvideo.avi')
    parser.add_argument('--file_path_recognition', type=str, default='./dataset/localizedLicensePlates')
    parser.add_argument('--output_path', type=str, default='./Output.csv')
    parser.add_argument('--sample_frequency', type=int, default=12)
    parser.add_argument('--save_files', type=bool, default=False)
    parser.add_argument('--stage', type=str, default='train_test_localization')
    args = parser.parse_args()
    return args


def get_localization_hyper_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast_stretch', type=float, default=0.7)
    parser.add_argument('--gaussian_blur_k', type=int, default=7)
    parser.add_argument('--gaussian_blur_sigma', type=float, default=1)
    parser.add_argument('--bifilter_k', type=int, default=11)
    parser.add_argument('--bifilter_sigma1', type=float, default=9)
    parser.add_argument('--bifilter_sigma2', type=float, default=15)
    parser.add_argument('--sharpen_k', type=int, default=11)
    parser.add_argument('--sharpen_sigma', type=float, default=1.25)
    parser.add_argument('--mask_low', type=object, default=[[12, 120, 110], [0, 0, 225], [0, 25, 45], [0, 125, 25], [0, 0, 0]])
    parser.add_argument('--mask_high', type=object, default=[[32, 255, 255], [180, 8, 255], [180, 90, 75], [180, 150, 100], [255, 255, 255]])
    parser.add_argument('--opening_kernel', type=tuple, default=(1, 2))
    parser.add_argument('--canny_lower', type=int, default=75)
    parser.add_argument('--canny_upper', type=int, default=245)
    parser.add_argument('--image_dim', type=tuple, default=(150, 50))
    parser.add_argument('--memoize_bounding_boxes', type=bool, default=True)
    parser.add_argument('--contour_ratio_epsilon', type=float, default=1.15)
    parser.add_argument('--contour_approximation_epsilon', type=float, default=0.035)
    parser.add_argument('--contour_perimeter', type=int, default=175)
    parser.add_argument('--center_offset_lookup', type=tuple, default=(100, 30))
    

    args = parser.parse_args()
    return args


def get_recognition_hyper_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast_stretch', type=object, default=[0.6, 0.75, 0.95])
    parser.add_argument('--hitmiss_kernel_1', type=tuple, default=(1, 20))
    parser.add_argument('--hitmiss_kernel_2', type=tuple, default=(1, 18))
    parser.add_argument('--hitmiss_kernel_3', type=tuple, default=(50, 1))
    parser.add_argument('--hitmiss_kernel_4', type=tuple, default=(45, 1))
    parser.add_argument('--opening_kernel_size', type=tuple, default=(2, 3))
    parser.add_argument('--sharpen_k', type=int, default=9)
    parser.add_argument('--sharpen_sigma', type=float, default=2)
    parser.add_argument('--sharpen_iter', type=int, default=4)
    parser.add_argument('--bifilter_k', type=int, default=9)
    parser.add_argument('--bifilter_sigma1', type=float, default=6)
    parser.add_argument('--bifilter_sigma2', type=float, default=15)
    parser.add_argument('--vertical_border_low_threshold', type=int, default=2)
    parser.add_argument('--min_char_jump', type=int, default=2)
    parser.add_argument('--horizontal_border_low_threshold', type=int, default=15)
    parser.add_argument('--horizontal_char_low_threshold', type=int, default=2)
    parser.add_argument('--char_segment_threshold', type=int, default=20)
    parser.add_argument('--vertical_ratio', type=float, default=3.5)
    parser.add_argument('--dash_range', type=int, default=3)
    parser.add_argument('--dash_threshold', type=int, default=14)
    parser.add_argument('--dash_size', type=int, default=14)
    parser.add_argument('--character_footprint_low', type=float, default=0.5)
    parser.add_argument('--character_footprint_high', type=float, default=0.9)
    parser.add_argument('--char_dist_threshold', type=int, default=2350)
    parser.add_argument('--plate_dist_threshold', type=int, default=21500)

    args = parser.parse_args()
    return args


# In this file, you need to pass three arguments into CaptureFrame_Process function.
if __name__ == '__main__':
    args = get_args()
    output_path = os.getcwd() + '/Output.csv' if args.output_path is None else args.output_path
    stage = args.stage
    if stage == "train_test_localization":
        cross_validation.cross_validation(args.file_path, 0, get_recognition_hyper_args())
    elif stage == "train_test_recognition":
        cross_validation.cross_validation(args.file_path_recognition, 1)
    elif stage == "test":
        CaptureFrame_Process.CaptureFrame_Process(
            args.file_path, args.sample_frequency,
            output_path, args.save_files,
            get_localization_hyper_args(),
            get_recognition_hyper_args())
