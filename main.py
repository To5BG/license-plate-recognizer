import argparse
import os
import CaptureFrame_Process
import numpy as np
from  crossValidation import cross_validation



# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./dataset/trainingsvideo.avi')
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--sample_frequency', type=int, default=2)
    parser.add_argument('--save_files', type=bool, default=False)
    parser.add_argument('--stage', type=str, default='train_test_recognition')
    args = parser.parse_args()
    return args


def get_localization_hyper_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast_stretch', type=float, default=0.7)
    parser.add_argument('--gaussian_blur_k', type=int, default=7)
    parser.add_argument('--gaussian_blur_sigma', type=float, default=1)
    parser.add_argument('--bifilter_k', type=int, default=11)
    parser.add_argument('--bifilter_sigma1', type=float, default=7)
    parser.add_argument('--bifilter_sigma2', type=float, default=15)
    parser.add_argument('--sharpen_k', type=int, default=11)
    parser.add_argument('--sharpen_sigma', type=float, default=1.5)
    parser.add_argument('--mask_low', type=object, default=[[10, 100, 100], [0, 0, 225], [0, 25, 45], [0, 125, 25], [0, 0, 0]])
    parser.add_argument('--mask_high', type=object, default=[[40, 255, 255], [180, 8, 255], [180, 90, 75], [180, 150, 100], [255, 255, 255]])
    parser.add_argument('--threshold_value', type=int, default=245)
    parser.add_argument('--opening_kernel', type=object, default=np.ones((1, 2)))
    parser.add_argument('--canny_lower', type=int, default=75)
    parser.add_argument('--canny_upper', type=int, default=200)
    parser.add_argument('--image_dim', type=tuple, default=(150, 50))
    parser.add_argument('--memoize_bounding_boxes', type=bool, default=True)
    parser.add_argument('--contour_ratio_epsilon', type=float, default=1.25)
    parser.add_argument('--contour_approximation_epsilon', type=float, default=0.03)
    
    args = parser.parse_args()
    return args

def get_recognition_hyper_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast_stretch', type=float, default=0.95)
    parser.add_argument('--hitmiss_kernel', type=object, default=np.ones((1, 20)))
    parser.add_argument('--opening_kernel_size', type=tuple, default=(3, 3))
    parser.add_argument('--vertical_border_low_threshold', type=int, default=3)
    parser.add_argument('--min_char_jump', type=int, default=3)
    parser.add_argument('--horizontal_border_low_threshold', type=int, default=25)
    parser.add_argument('--horizontal_char_low_threshold', type=int, default=4)
    parser.add_argument('--char_segment_threshold', type=int, default=6)
    parser.add_argument('--sharpen_k', type=int, default=11)
    parser.add_argument('--sharpen_sigma', type=float, default=1.5)
    parser.add_argument('--bifilter_k', type=int, default=11)
    parser.add_argument('--bifilter_sigma1', type=float, default=7)
    parser.add_argument('--bifilter_sigma2', type=float, default=15)
    
    args = parser.parse_args()
    return args

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
    stage = args.stage
    if stage == "train_test_localization":
        cross_validation(file_path, get_localization_hyper_args(), 0)
    elif stage  == "train_test_recognition":
        cross_validation("dataset/localizedLicensePlates", get_recognition_hyper_args(), 1)
    elif stage == "test":
        CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path, save_files, get_localization_hyper_args(), get_recognition_hyper_args())
