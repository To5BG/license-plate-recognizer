import cv2
import os
import pandas as pd
import Localization
import Recognize

i = 1
lastGuess = None

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""


def CaptureFrame_Process(file_path, sample_frequency, save_path, saveFiles, localization_hyper_args,
                         recognition_hyper_args):
    vid = cv2.VideoCapture(file_path)
    # Check if camera opened successfully

    if (vid.isOpened() == False):
        print("Error opening video stream or file")

    # Create image folder is saveFiles is True
    cwd = os.path.abspath(os.getcwd())
    if saveFiles and not os.path.exists(os.path.join(cwd, "images")):
        os.makedirs(os.path.join(cwd, "images"))
    # Keep track of frame count
    frame_count = 0
    # Calculate sampling rate based on frame count
    fps = vid.get(cv2.CAP_PROP_FPS)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar_rate = total // 100
    rate = fps // sample_frequency
    data = {}
    cache = [{}, {}]
    last_frame = None
    fc = 0
    # Read until video is completed
    while (vid.isOpened()):
        # Capture frame-by-frame based on sampling frequency
        succ, frame = vid.read()
        if not succ:
            break
        if frame_count % rate == 0:
            if saveFiles:
                cv2.imwrite(os.path.join(cwd, "images", "frame%d.jpg" % frame_count), frame)
            # Localize and recognize plates
            plates, _ = Localization.plate_detection(frame, localization_hyper_args, recognition_hyper_args)
            plate_nums = Recognize.segment_and_recognize(plates, recognition_hyper_args)

            # Majority voting -> have a cache and save only the most common license plate in the output.csv
            # Triggered if the scene is different
			# Match template moves the image pattern through the template to find the best match of the image inside the template
    		# In the case of the two images having the same shape, as here, the result is simply a normalized euclidean distance between the two images
    		# Hence a separate implementation was deemed unecessary
    		# https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be
            if last_frame is None or cv2.matchTemplate(last_frame, frame, 1) > 0.2:
                pic = last_frame
                last_frame = frame

                cache, data = majorityVote(cache, pic, data, frame, fc, fps, cwd)  # resets or continues with cache

                fc = frame_count  # next frame which to put in csv
            for i, plate_num in enumerate(plate_nums):
                if plate_num != '':
                    cache[i][plate_num] = cache[i].get(plate_num, 0) + 1  # increment vote for given match
            if frame_count % progress_bar_rate == 0:
                updateProgressBar(frame_count, total)
        frame_count += 1
    # Finalize progress bar
    updateProgressBar(frame_count, total)
    # Add last entry
    for c in cache:
        maj = max(c, key=c.get, default=None)
        if maj is not None and maj not in data.keys():
            data[maj] = (maj, fc, round(fc / fps, 5))
    # Save csv
    pd.DataFrame(list(data.values()), columns=['License plate', 'Frame no.', 'Timestamp(seconds)']).to_csv(save_path,
                                                                                                           index=False)
    # When everything done, release the video capture object
    vid.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def majorityVote(cache, pic, data, frame, fc, fps, cwd):
    global i, lastGuess
    totalVotes = sum([val for d in cache for val in d.values()])

    for c in cache:

        maj = max(c, key=c.get, default=None)

        # if there is any guess take the most common one and put it in the data for the output.csv file

        if (lastGuess is not None and maj is not None):
            l_distance = levenshtein_distance(maj, lastGuess)

            # print(l_distance)
            # print(maj)
            # print(lastGuess)
            # print(cv2.matchTemplate(pic, frame, 1))
            if l_distance <= 3 and l_distance > 0 or totalVotes <= 15:
                lastGuess = maj
                return cache, data
        if maj is not None and maj not in data.keys():
            print("---------------")
            print(totalVotes)
            print(str(i) + "_" + maj + " _" + str(fc) + ".jpg", "\n-------------")
            cv2.imwrite(os.path.join(cwd, "debug", str(i) + "_" + maj + "_" + str(fc) + ".jpg"), pic)
            i += 1
            data[maj] = (maj, fc, round(fc / fps, 5))
            lastGuess = maj


    return [{}, {}], data


def levenshtein_distance(plate1, plate2):
    # Create an empty matrix to hold the distance values
    m = len(plate1)
    n = len(plate2)
    distance = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill in the first row and column of the matrix
    for i in range(m + 1):
        distance[i][0] = i
    for j in range(n + 1):
        distance[0][j] = j

    # Iterate over the matrix, computing the distance values
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If the characters match, the distance is the same as the previous diagonal value
            if plate1[i - 1] == plate2[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                # Otherwise, the distance is the minimum of the three possible operations (insertion, deletion, substitution)
                distance[i][j] = min(distance[i - 1][j], distance[i][j - 1], distance[i - 1][j - 1]) + 1

    # The distance between the two strings is the value in the bottom-right corner of the matrix
    return distance[m][n]


def updateProgressBar(curr, total):
    percent = ("{0:.1f}").format(100 * (curr / float(total)))
    filledLength = int(50 * curr // total)
    bar = '█' * filledLength + '-' * (50 - filledLength)
    print(f'\r{"Progress:"} |{bar}| {percent}% {"Complete"}', end="\r")
