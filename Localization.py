import cv2
import numpy as np
from Recognize import segment_and_recognize

last_image = None
last_boxes = list()
last_plate_imgs = list()
desired_color_range = None

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""


def plate_detection(image, hyper_args, quick_rec_hyper_args, debug=False):
    # ---------------------------------------
    # ------- STAGE 1 - PREPROCESING --------
    # ---------------------------------------

    global last_image, desired_color_range, last_boxes, last_plate_imgs
    # Guard clause for first frame
    if last_image is None: last_image = image
    # If new frame (not similar to last one), set new last_image and reset color range
    # Match template moves the image pattern through the template to find the best match of the image inside the template
    # In the case of the two images having the same shape, as here, the result is simply a normalized euclidean distance between the two images
    # Hence a separate implementation was deemed unecessary
    # https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be
    if cv2.matchTemplate(image, last_image, 1) > 0.2:
        last_image = image
        desired_color_range = None
        last_boxes = list()
        last_plate_imgs = list()

    # Contrast stretch the image
    img = image.copy()
    if hyper_args.contrast_stretch != 0.0:
        img = contrastImprovementContrastStretching(img, hyper_args.contrast_stretch, 0, 255)
    # Blur to remove noise
    img = cv2.GaussianBlur(img, (hyper_args.gaussian_blur_k, hyper_args.gaussian_blur_k),
                           hyper_args.gaussian_blur_sigma)
    img = cv2.bilateralFilter(img, hyper_args.bifilter_k, hyper_args.bifilter_sigma1, hyper_args.bifilter_sigma2)
    # Sharpen edges
    img = cv2.filter2D(img, -1, sharpKernel(hyper_args.sharpen_k, hyper_args.sharpen_sigma))

    # ---------------------------------------
    # ------- STAGE 2 - SEGMENTATION --------
    # ---------------------------------------

    # Color segmentation
    # Go through every range until enough license plates are found
    for ml, mh in zip(hyper_args.mask_low, hyper_args.mask_high):

        # If already have preferred color range and does not match current frame, skip iteration
        if desired_color_range is not None and (
                set(desired_color_range[0]) != set(ml) and set(desired_color_range[1]) != set(mh)): continue

        # Define color range
        # Similar to Lab_1_Color_And_Histograms color segmentation
        colorMin = np.array(ml)
        colorMax = np.array(mh)
        # Segment only the selected color from the image and leave out all the rest (apply a mask)
        mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), colorMin, colorMax)
        plate_imgs = cv2.bitwise_and(img, img, mask=mask)

        # ---------------------------------------
        # ------ STAGE 3 - CANNY & MORPH --------
        # ---------------------------------------

        # Using cv2's implementation directly - approved from Lab_4_Isodata_Edge_Detection
        ## Canny's algorithm for edge detection
        edged = cv2.Canny(plate_imgs, hyper_args.canny_lower, hyper_args.canny_upper)

        # Using cv2's morph interface directly - approved from Lab_2_Morphology
        ## Noise reduction after finding segmentation
        # Using morphological filtering
        if set(ml) == set([0, 0, 0]) or set(mh) == set([255, 255, 255]):
            edged_horizontal = cv2.morphologyEx(edged, cv2.MORPH_DILATE, np.ones((1, 2)), iterations=1)
            edged_vertical = cv2.morphologyEx(edged, cv2.MORPH_DILATE, np.ones((2, 1)), iterations=1)
            edged = cv2.bitwise_or(edged_horizontal, edged_vertical)
        else:
            edged_horizontal = cv2.morphologyEx(edged, cv2.MORPH_OPEN, np.ones(hyper_args.opening_kernel), iterations=1)
            edged_vertical = cv2.morphologyEx(edged, cv2.MORPH_OPEN, np.ones(tuple(reversed(hyper_args.opening_kernel))), iterations=1)
            edged = cv2.bitwise_or(edged_horizontal, edged_vertical)

            edged_horizontal = cv2.morphologyEx(edged, cv2.MORPH_DILATE, np.ones((1, 2)), iterations=4)
            edged_vertical = cv2.morphologyEx(edged, cv2.MORPH_DILATE, np.ones((2, 1)), iterations=4)
            edged = cv2.bitwise_or(edged_horizontal, edged_vertical)

        if debug: cv2.imshow("Edges detected", edged)

        # ---------------------------------------
        # --- STAGE 4 - CONTOUR / BOUNDING ------
        # ---------------------------------------

        # Find bounding boxes
        boxes = list()
        centers = list()
        plate_imgs = list()
        # By using contour detection
        # Using cv2's contour detection implementation directly - approved from Lab_6_Find_Contours_SIFT
        cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Get only some of the contours
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[
               :(min(len(cnts), 100) if ml is None or mh is None else min(len(cnts), 20))]
        for cnt in cnts:
            # Keep perimeter for later check and approximation
            peri = cv2.arcLength(cnt, True)
            # Approximate polygon, given a contour, with some epsilon for roughness
            approx = cv2.approxPolyDP(cnt, hyper_args.contour_approximation_epsilon * peri, True)
            rect = cv2.minAreaRect(cnt)
            if rect[1][0] == 0 or rect[1][1] == 0: continue
            # OpenCV may at times consider 90 deg rotation with flipped width/height equivalent to regular poly
            ratio = rect[1][0] / rect[1][1] if rect[2] <= 45 else rect[1][1] / rect[1][0]  # max(rect[1][0], rect[1][1]) / min(rect[1][0], rect[1][1])
            # Checked conditions for a license plate:
            # - Check that contour approximates a quadilateral
            # - Check that ratio of said quad is some epsilon away from 4.5, the most common license plate ratio
            # - Check that perimeter is large enough to be considered a plate (project desc guarantees 100 plate width, hence at least 2 times that)
            # - Check that said contour does not approximate already checked area, by comparing centers (100 pixels on x, 30 on y)
            if (
                    len(approx) == 4 and
                    abs(ratio - 4.5) <= hyper_args.contour_ratio_epsilon and
                    peri > hyper_args.contour_perimeter and
                    (len(centers) == 0 or next(
                        filter(lambda c: abs(c[0] - rect[0][0]) > hyper_args.center_offset_lookup[0] 
                            or abs(c[1] - rect[0][1]) > hyper_args.center_offset_lookup[1], centers), None) is not None)
            ):
                # ---------------------------------------
                # --- STAGE 5 - WARPING / CROPPING ------
                # ---------------------------------------

                # OpenCV may at times consider 90 deg rotation with flipped width/height equivalent to regular poly
                # Hence the more complicated logic
                rot = rect[2] if rect[2] < 45 else rect[2] - 90
                # Rotate image to make license plate x-axis aligned
                rot_img = cv2.warpAffine(image, cv2.getRotationMatrix2D(rect[0], rot, 1),
                                         (image.shape[1], image.shape[0]))
                # Crop and resize plate
                rot_img = cv2.getRectSubPix(rot_img, (int(rect[1][0]), int(rect[1][1])) if rect[2] < 45 else (
                int(rect[1][1]), int(rect[1][0])), tuple(map(int, rect[0])))
                resized_img = cv2.resize(rot_img, hyper_args.image_dim)
                # If plate is too disimilar with other plate imgs, consider as noise
                if 'F' in segment_and_recognize([resized_img], quick_rec_hyper_args, debug=False, quick_check=True): continue
                # Store results
                centers.append(rect[0])
                box = cv2.boxPoints(rect)
                boxes.append(np.array(box).astype(np.int32))
                if ml != hyper_args.mask_low[-1] and mh != hyper_args.mask_high[-1]:
                    desired_color_range = (ml, mh)
                plate_imgs.append(resized_img)

        if len(boxes) != 0: break

    # ---------------------------------------
    # -------- STAGE X - DEFAULTING ---------
    # ---------------------------------------

    # More logic used to return the old stored license plate if no contours are found for current frame
    # (With of course, checking that frame has remained mostly the same)
    if hyper_args.memoize_bounding_boxes:
        # If less plates are found, set to previous plates
        if len(boxes) < len(last_boxes):
            boxes = last_boxes
            plate_imgs = last_plate_imgs
        # Else rewrite last boxes and image
        else:
            last_image = image
            last_boxes = boxes
            last_plate_imgs = plate_imgs

    indices = np.argsort([b[0][0] for b in boxes])
    plate_imgs = list([plate_imgs[i] for i in indices])
    isDutch = desired_color_range == ((hyper_args.mask_low[0], hyper_args.mask_high[0])
        or desired_color_range == (hyper_args.mask_low[1], hyper_args.mask_high[1]))
    return plate_imgs, boxes, isDutch


# Approved from Lab_1_Color_And_Histograms
def contrastImprovementContrastStretching(img, contrastFactor, minV, maxV):
    # Guard clause
    q = max(min(contrastFactor, 1), 0)
    for c in range(img.shape[2]):
        # Calculate global lows and highs (quantiles for percentile stretch, returns min and max (regular stretch) if f = 1.0)
        low = np.quantile(img[:, :, c], (1 - q) / 2)
        high = np.quantile(img[:, :, c], 1 - (1 - q) / 2)
        img[:, :, c] = np.clip((img[:, :, c].astype(np.float16) - low) * (maxV - minV) / (high - low) + minV, 0.0,
                               255.0).astype(int)
    return img


# Approved from Lab_3_Filtering
def sharpKernel(kernel, sigma):
    k = (kernel - 1) // 2
    sharpen_kernel = None
    # Do box filtered sharpen kernel instead
    if sigma < 0.0 or sigma > kernel:
        sharpen_kernel = np.ones((kernel, kernel)) / - kernel ** 2
        sharpen_kernel[k, k] = 2 - 1 / kernel ** 2
    else:
        sharpen_kernel = np.zeros((kernel, kernel))
        sharpen_kernel[k, k] = 2
        g = cv2.getGaussianKernel(kernel, sigma)
        sharpen_kernel -= g @ g.T
    return sharpen_kernel
