from cross_validation_localization import cross_validate as cvl
from cross_validation_recognition import cross_validate as cvr

def cross_validation(file_path, test_stage, rec_quick_args=None):
    if test_stage == 0:
        cvl(file_path, get_localization_hyper_args(), rec_quick_args)
    if test_stage == 1:
        cvr(file_path, get_recognition_hyper_args())

def get_localization_hyper_args():
    args = {}
    args["contrast_stretch"] = [0.7]
    args["gaussian_blur_k"] = [7]
    args["gaussian_blur_sigma"] = [1]
    args["bifilter_k"] = [11]
    args["bifilter_sigma1"] = [7]
    args["bifilter_sigma2"] = [15]
    args["sharpen_k"] = [11]
    args["sharpen_sigma"] = [1.5]
    args["mask_low"] = [[[10, 100, 100], [0, 0, 225], [0, 25, 45], [0, 125, 25], [0, 0, 0]]]
    args["mask_high"] = [[[40, 255, 255], [180, 8, 255], [180, 90, 75], [180, 150, 100], [255, 255, 255]]]
    args["threshold_value"] = [245]
    args["opening_kernel"] = [(1, 2)]
    args["canny_lower"] = [75]
    args["canny_upper"] = [200]
    args["image_dim"] = [(150, 50)]
    args["memoize_bounding_boxes"] = [True]
    args["contour_ratio_epsilon"] = [1.25]
    args["contour_approximation_epsilon"] = [0.03]
    args["contour_perimeter"] = [200]
    args["center_offset_lookup"] = [(100, 30)]
    return args

def get_recognition_hyper_args():
    args = {}
    args["contrast_stretch"] = [[0.6, 0.75, 0.95]]
    args["hitmiss_kernel_1"] = [(1, 25)]
    args["hitmiss_kernel_2"] = [(1, 20)]
    args["hitmiss_kernel_3"] = [(50, 1)]
    args["hitmiss_kernel_4"] = [(45, 1)]
    args["opening_kernel_size"] = [(3, 3)]
    args["sharpen_k"] = [9]
    args["sharpen_sigma"] = [2]
    args["sharpen_iter"] = [4]
    args["bifilter_k"] = [9]
    args["bifilter_sigma1"] = [6]
    args["bifilter_sigma2"] = [15]
    args["vertical_border_low_threshold"] = [2]
    args["min_char_jump"] = [2]
    args["horizontal_border_low_threshold"] = [15]
    args["horizontal_char_low_threshold"] = [2]
    args["char_segment_threshold"] = [20]
    args["vertical_ratio"] = [3.5]
    args["dash_range"] = [5]
    args["dash_threshold"] = [12]
    args["dash_size"] = [12]
    args["character_footprint_low"] = [0.5]
    args["character_footprint_high"] = [0.9]
    args["char_dist_threshold"] = [2350]
    args["plate_dist_threshold"] = [21500]
    return args