import argparse
import os
import CaptureFrame_Process
import numpy as np


# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default='./dataset/trainingsvideo.avi')
	parser.add_argument('--output_path', type=str, default='./Output.csv')
	parser.add_argument('--sample_frequency', type=int, default=2)
	parser.add_argument('--save_files', type=bool, default=False)
	args = parser.parse_args()
	return args

def get_hyper_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--contrast_stretch', type=float, default=0.75)
	parser.add_argument('--gaussian_blur_k', type=tuple, default=(5,5))
	parser.add_argument('--gaussian_blur_sigma', type=float, default=0.75)
	parser.add_argument('--mask_low', type=object, default=[15,135,135])
	parser.add_argument('--mask_high', type=object, default=[40,255,255])
	parser.add_argument('--threshold_value', type=int, default=50)
	parser.add_argument('--opening_kernel', type=object, default=np.ones((3, 3)))
	parser.add_argument('--hitmiss_kernel', type=object, default=np.ones((2, 5)))
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
	CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path, save_files, get_hyper_args())