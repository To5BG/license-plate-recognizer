import cv2
import os
import pandas as pd
import Localization
import Recognize


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
def CaptureFrame_Process(file_path, sample_frequency, save_path, saveFiles, localization_hyper_args, recognition_hyper_args):

	vid = cv2.VideoCapture(file_path)
	# Check if camera opened successfully

	if (vid.isOpened()== False): 
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
	while(vid.isOpened()):
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
			if last_frame is None or cv2.matchTemplate(last_frame, frame, 1) > 0.2:
				last_frame = frame
				for c in cache:
					maj = max(c, key=c.get, default=None)
					if maj is not None and maj not in data.keys():
						data[maj] = (maj, fc, round(fc / fps, 5))
				cache = [{}, {}]
				fc = frame_count
			for i, plate_num in enumerate(plate_nums):
				if plate_num != '':
					cache[i][plate_num] = cache[i].get(plate_num, 0) + 1
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
	pd.DataFrame(list(data.values()), columns=['License plate', 'Frame no.', 'Timestamp(seconds)']).to_csv(save_path, index=False)
	# When everything done, release the video capture object
	vid.release()
	# Closes all the frames
	cv2.destroyAllWindows()

def updateProgressBar(curr, total):
	percent = ("{0:.1f}").format(100 * (curr / float(total)))
	filledLength = int(50 * curr // total)
	bar = '█' * filledLength + '-' * (50 - filledLength)
	print(f'\r{"Progress:"} |{bar}| {percent}% {"Complete"}', end = "\r")