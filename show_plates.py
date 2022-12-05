import argparse
from Localization import plate_detection
import cv2 
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./dataset/dummytestvideo.avi')
    args = parser.parse_args()
    return args 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(get_args().file_path)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    detections, border = plate_detection(frame)
    # Add predicted box
    frame = cv2.polylines(frame, [border], True, (255, 0, 0), 3)
    # Display the resulting frame
    cv2.imshow('Frame', detections)
    # Press P on keyboard to pause
    if cv2.waitKey(24) == ord('p'):
      while (True):
        if cv2.waitKey(24) == ord('p'):
          break
    # Press Q on keyboard to exit
    if cv2.waitKey(24) == ord('q'):
      break
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()



