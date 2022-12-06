import argparse
from Localization import plate_detection
from main import get_hyper_args
import cv2 
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='dataset\TrainingSet\Categorie I\Video2_2.avi')
    args = parser.parse_args()
    return args 

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(get_args().file_path)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

playbackSpeed = 10
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret != True: break;
  detections, borders = plate_detection(frame, get_hyper_args())
  for border in borders:
    # Add predicted box
    frame = cv2.polylines(frame, [border], True, (255, 0, 0), 3)
  # Display the resulting frame
  cv2.namedWindow('Full frame', cv2.WINDOW_NORMAL)
  cv2.imshow('Full frame', frame)

  cv2.imshow('Cropped plates', detections)

  a = cv2.waitKey(playbackSpeed)
  # Press P on keyboard to pause
  if a == ord('p'):
    while (True):
      if cv2.waitKey(playbackSpeed) == ord('p'):
        break
  # Press Q on keyboard to exit
  if a == ord('q'):
    break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()



