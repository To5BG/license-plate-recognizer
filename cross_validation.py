from cross_validation_localization import cross_validate as cvl
from cross_validation_recognition import cross_validate as cvr
import argparse
import numpy as np

def cross_validation(file_path, test_stage, rec_quick_args=None):
    if test_stage == 0:
        cvl(file_path, get_localization_hyper_args(), rec_quick_args)
    if test_stage == 1:
        cvr(file_path, get_recognition_hyper_args())

def get_localization_hyper_args():
    args = {}
    args["contrast_stretch"] = [0.5, 0.6, 0.75, 0.85, 0.95, 0.98, 1]
    args["gaussian_blur_k"] = [3, 5, 7, 9, 11]
    args["gaussian_blur_sigma"] = [0.5, 0.75, 1, 1.25, 1.5]
    args["bifilter_k"] = [9, 11, 13, 15]
    args["bifilter_sigma1"] = [7, 9, 11, 13, 15]
    args["bifilter_sigma2"] = [11, 13, 15, 17, 19]
    args["sharpen_k"] = [7, 9, 11, 13, 15]
    args["sharpen_sigma"] = [0.5, 0.75, 1, 1.25, 1.5]
    args["mask_low"] = [[[10, 100, 100], [0, 0, 225], [0, 25, 45], [0, 125, 25], [0, 0, 0]]]
    args["mask_high"] = [[[40, 255, 255], [180, 8, 255], [180, 90, 75], [180, 150, 100], [255, 255, 255]]]
    args["threshold_value"] = [220, 225, 230, 235, 240, 245, 250]
    args["opening_kernel"] = [np.ones((1, 2)), np.ones((1, 3)), np.ones((1, 5))]
    args["canny_lower"] = [55, 65, 75, 90, 100]
    args["canny_upper"] = [175, 185, 200, 210, 225]
    args["image_dim"] = [(150, 50)]
    args["memoize_bounding_boxes"] = [True]
    args["contour_ratio_epsilon"] = [1, 1.15, 1.25, 1.35, 1.5]
    args["contour_approximation_epsilon"] = [0.02, 0.025, 0.03, 0.035, 0.04]
    return args

def get_recognition_hyper_args():
    args = {}
    args["contrast_stretch"] = [0.64]
    args["hitmiss_kernel_1"] = [(1, 25)]
    args["hitmiss_kernel_2"] = [(1, 20)]
    args["hitmiss_kernel_3"] = [(50, 1)]
    args["hitmiss_kernel_4"] = [(45, 1)]
    args["opening_kernel_size"] = [(3, 3)]
    args["sharpen_k"] = [11]
    args["sharpen_sigma"] = [1.5]
    args["sharpen_iter"] = [3]
    args["bifilter_k"] = [11]
    args["bifilter_sigma1"] = [7]
    args["bifilter_sigma2"] = [15]
    args["vertical_border_low_threshold"] = [3]
    args["min_char_jump"] = [3]
    args["horizontal_border_low_threshold"] = [20]
    args["horizontal_char_low_threshold"] = [3]
    args["char_segment_threshold"] = [20]
    args["dash_threshold"] = [0.85]
    args["char_dist_threshold"] = [2250]
    args["plate_dist_threshold"] = [22500]
    return args