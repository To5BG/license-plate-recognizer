import argparse
from Localization import plate_detection
from main import get_hyper_args
import cv2 
import numpy as np
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='dataset/dummytestvideo.avi')
    args = parser.parse_args()
    return args 

header = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'frame', 'time']
def captureBox(event, x, y, flags, param):
  global mbp, frame_count, fps, addEntry
  if event == cv2.EVENT_LBUTTONUP:
    # If 4 points are accrued, add new box entry
    if len(mbp) == 4:
      # Flatten point tuples for csv entry
      e = []
      [e.extend([t[0], t[1]]) for t in mbp]
      e.extend([frame_count, frame_count / fps])
      with open('BoundingBoxGroundTruth.csv', 'r+') as f:
        df = f.readlines()
        # If file is empty, add header
        if (len(df) == 0):
          f.write(header)
        # Start from original csvLine for easier time finding the entry
        idx = csvLine
        # Move idx up and down to find the place of update/insertion
        while (int(df[idx].split(",")[8]) > frame_count):
          idx -= 1
        while (int(df[idx].split(",")[8]) < frame_count):
          idx += 1
        # Overwrite decision if csv has no entries (but header), or if idx is at the last row
        idx = min(len(df) - 1, max(1, idx))
        # If an entry already exists for the given frame count, update it
        if int(df[idx].split(",")[8]) == frame_count and not addEntry:
          df[idx] = ",".join(map(str, e)) + "\n"
        # Otherwise insert a new entry for this frame
        else:
          df.insert(idx, ",".join(map(str, e)) + "\n")
        # Write new df
        f.seek(0)
        f.writelines(df)
      mbp = []
  elif event == cv2.EVENT_LBUTTONDOWN:
    # Add point to entry
    mbp.append((x, y))

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(get_args().file_path)
fps = cap.get(cv2.CAP_PROP_FPS)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

playbackSpeed = 10
# Read csv ground truth file
groundTruthBoxes = open("BoundingBoxGroundTruth.csv", "r").read().split('\n')
# Variable to keep track of current csvLine
csvLine = 0
# Next recorded frame
nextFrame = -1 if groundTruthBoxes[csvLine + 1] == '' else int(groundTruthBoxes[csvLine + 1].split(',')[-2])
# On manual select mode - add entry instead of overwriting (for more than 1 license plate on an image)
addEntry = False
# Frame count
frame_count = 0
# Manual Box Points
mbp = []
# Boolean flag for manual framing
usedCaretForNextFrame = False
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if not ret: break
  detections, borders = plate_detection(frame, get_hyper_args())
  if nextFrame == frame_count:
    pointarr = list()
    while nextFrame == frame_count:
      csvLine += 1
      nextFrame = -1 if groundTruthBoxes[csvLine + 1] == '' else int(groundTruthBoxes[csvLine + 1].split(',')[-2])
      pointarr.append(np.array([[int(a), int(b)] for a,b in zip(groundTruthBoxes[csvLine].split(',')[0:8:2], groundTruthBoxes[csvLine].split(',')[1:8:2])]))
  # Add predicted box
  for border in borders:
    frame = cv2.polylines(frame, [border], True, (255, 0, 0), 3)
  # Add ground truth box, new var for easier onclick event handling
  for points in pointarr:
    frame = cv2.polylines(frame, [points], True, (0, 255, 0), 3)

  # Display the original frame with bounding boxes
  cv2.namedWindow('Original frame', cv2.WINDOW_NORMAL)
  cv2.setMouseCallback('Original frame', captureBox)
  cv2.imshow('Original frame', frame)
  
  # Display cropped plates
  for j, plate in enumerate(detections):
    cv2.imshow('Cropped plate #%d' % j, detections[j])

  a = cv2.waitKey(playbackSpeed)
  # Press P on keyboard to pause
  if usedCaretForNextFrame or a == ord('p'):
    usedCaretForNextFrame = False
    while (True):
      a = cv2.waitKey(playbackSpeed)
      if a == ord('n'):
        usedCaretForNextFrame = True
        break
      if a == ord('m'):
        addEntry = not addEntry
        msg = "Entries of same frame are now added" if addEntry else "Entries of same frame are now overriden"
        print(msg)
      if a == ord('p'):
        break
  # Press Q on keyboard to exit
  if a == ord('q'):
    break
  frame_count += 1

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
