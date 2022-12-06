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
def CaptureFrame_Process(file_path, sample_frequency, save_path, saveFiles, hyper_args):
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
	rate = fps // sample_frequency
	data = {}
	# Read until video is completed
	while(vid.isOpened()):
		# Capture frame-by-frame based on sampling frequency
		succ, frame = vid.read()
		if not succ:
			break
		if frame_count % rate == 0:
			if saveFiles:
				cv2.imwrite(os.path.join(cwd, "images", "frame%d.jpg" % frame_count), frame)
			plates, _ = Localization.plate_detection(frame, hyper_args)
			plate_numbers = Recognize.segment_and_recognize(plates)
			for pnum in plate_numbers:
				#### ASSUMES LICENSE PLATES DO NOT REPEAT
				#### ALSO WON'T ALLOW FOR MULTI-FRAME VALIDATION
				#### CHANGE LATER
				if pnum not in data:
					data[pnum] = (pnum, frame_count, round(frame_count / fps, 5))
		frame_count += 1
	# Save csv
	pd.DataFrame(list(data.values()), columns=['License plate', 'Frame no.', 'Timestamp(seconds)']).to_csv(
		os.path.join(save_path, "Output.csv"), index=False)
	# When everything done, release the video capture object
	vid.release()
	# Closes all the frames
	cv2.destroyAllWindows()
