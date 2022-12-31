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
    parser.add_argument('--stage', type=str, default='train_test')
    args = parser.parse_args()
    return args


def get_hyper_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast_stretch', type=float, default=0.7)
    parser.add_argument('--gaussian_blur_k', type=int, default=7)
    parser.add_argument('--gaussian_blur_sigma', type=float, default=1)
    parser.add_argument('--bifilter_k', type=int, default=11)
    parser.add_argument('--bifilter_sigma1', type=float, default=7)
    parser.add_argument('--bifilter_sigma2', type=float, default=15)
    parser.add_argument('--sharpen_k', type=int, default=11)
    parser.add_argument('--sharpen_sigma', type=float, default=1.5)
    parser.add_argument('--mask_low', type=object, default=[10, 115, 115])
    parser.add_argument('--mask_high', type=object, default=[40, 255, 255])
    parser.add_argument('--threshold_value', type=int, default=245)
    parser.add_argument('--opening_kernel', type=object, default=np.ones((1, 2)))
    parser.add_argument('--hitmiss_kernel', type=object, default=np.ones((1, 2)))
    parser.add_argument('--canny_lower', type=int, default=75)
    parser.add_argument('--canny_upper', type=int, default=200)
    parser.add_argument('--image_width', type=int, default=150)
    parser.add_argument('--memoize_bounding_boxes', type=bool, default=True)
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
    if stage == "train_test":
        cross_validation(file_path, get_hyper_args())
    elif stage  == "train":
        pass
    elif stage == "test":
        CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path, save_files, get_hyper_args())
