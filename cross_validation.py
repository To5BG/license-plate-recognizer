from cross_validation_localization import cross_validate as cvl
from cross_validation_recognition import cross_validate as cvr


def cross_validation(file_path, test_stage, rec_quick_args=None):
    if test_stage == 0:
        cvl(file_path, get_localization_hyper_args(), rec_quick_args)
    if test_stage == 1:
        cvr(file_path, get_recognition_hyper_args())


def get_localization_hyper_args():
    args = {}
    low_mask_permutations = []
    for h in range(12, 15):
        for s in range(110, 131):
            for v in range(100, 121):
                low_mask_permutations.append([[h, s, v]])
    high_mask_permutations = []
    for h in range(31, 34):
        for s in [255]:
            for v in [255]:
                high_mask_permutations.append([[h, s, v]])
    args["contrast_stretch"] = [0.7]
    args["gaussian_blur_k"] = [7]
    args["gaussian_blur_sigma"] = [1]
    args["bifilter_k"] = [11]
    args["bifilter_sigma1"] = [9]
    args["bifilter_sigma2"] = [15]
    args["sharpen_k"] = [11]
    args["sharpen_sigma"] = [1.25]
    args["mask_low"] = [[[12, 120, 110],  [0, 0, 0]]] #low_mask_permutations
    args["mask_high"] = [[[32, 255, 255], [255, 255, 255]]] #high_mask_permutations
    args["opening_kernel"] = [(1, 2)]
    args["canny_lower"] = [75]
    args["canny_upper"] = [245]
    args["image_dim"] = [(150, 50)]
    args["memoize_bounding_boxes"] = [False]
    args["contour_ratio_epsilon"] = [1.15]
    args["contour_approximation_epsilon"] = [0.035]
    args["contour_perimeter"] = [175]
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
    args["horizontal_char_low_threshold"] = [19]
    args["char_segment_threshold"] = [1.5]
    args["vertical_ratio"] = [3.5]
    args["dash_range"] = [1,2,3]
    args["dash_threshold"] = [5,6,7,8,9]
    args["dash_size"] = [5,6,7,8,9,10, 11, 12, 13, 14]
    args["character_footprint_low"] = [0.25]
    args["character_footprint_high"] = [0.9]
    args["char_dist_threshold"] = [1650]
    args["plate_dist_threshold"] = [26000]

    return args
