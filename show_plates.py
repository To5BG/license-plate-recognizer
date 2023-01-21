import argparse
from Localization import plate_detection
from Recognize import segment_and_recognize
from main import get_localization_hyper_args, get_recognition_hyper_args
import cv2 
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--file_path', type=str, default='dataset/TrainingSet/Categorie I/Video7_2.avi')
    parser.add_argument('--file_path', type=str, default='dataset/dummytestvideo.avi')
    parser.add_argument('--stage', type=int, default=1)
    args = parser.parse_args()
    return args 

# Click event for creating ground truth bounding boxes for localization crossval
header = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'frame', 'time']
def captureBoxEvent(event, x, y, flags, param):
  global mbp, frame_count, fps, addEntry
  if event == cv2.EVENT_LBUTTONUP:
    # If 4 points are accrued, add new box entry
    if len(mbp) == 4:
      # Flatten point tuples for csv entry
      e = []
      [e.extend([t[0], t[1]]) for t in mbp]
      frame_count -= 1
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
        frame_count += 1
        # Write new df
        f.seek(0)
        f.writelines(df)
      mbp = []
  elif event == cv2.EVENT_LBUTTONDOWN:
    # Add point to entry
    mbp.append((x, y))

# Click event for cropping plates for recognition crossval
def cropPlateEvent(event, x, y, flags, param):
  global mcp, frame, frame_count, cwd
  if event == cv2.EVENT_LBUTTONUP:
    # If 4 points are accrued, add new box entry
    if len(mcp) == 4:
      # Save plate as file
      box = cv2.minAreaRect(np.float32(mcp))
      rot = box[2] if box[2] < 45 else box[2] - 90
      # Rotate image to make license plate x-axis aligned
      rot_img = cv2.warpAffine(frame, cv2.getRotationMatrix2D(box[0], rot, 1),
                                (frame.shape[1], frame.shape[0]))
      # Crop and resize plate
      rot_img = cv2.getRectSubPix(rot_img, (int(box[1][0]), int(box[1][1])) if box[2] < 45 else (
      int(box[1][1]), int(box[1][0])), tuple(map(int, box[0])))
      resized_img = cv2.resize(rot_img, get_localization_hyper_args().image_dim)
      cv2.imwrite(os.path.join(cwd, "dataset", "localizedLicensePlates", "frame%drecdata.jpg" % frame_count), resized_img)
      mcp = []
  elif event == cv2.EVENT_LBUTTONDOWN:
    # Add point to entry
    mcp.append(np.float32([x, y]))


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
# Toggle between data collection for localization and recognition (bounding box vs crop plate)
cropPlate = False
# Frame count
frame_count = 0
# Decremental counter for quick-skipping frames
skipFrames = 0
# Manual Box points for loc
mbp = []
# Manual points for rec
mcp = [] 
# Frame to be used for cropping in the rec click event
frame = None
# Current working dir path
cwd = os.path.abspath(os.getcwd())
# Boolean flag for manual framing
usedCaretForNextFrame = False
# Set up widnow with default click event
cv2.namedWindow('Original frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Original frame', captureBoxEvent)
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if not ret: break
  if nextFrame == frame_count:
    pointarr = list()
    while nextFrame == frame_count:
      csvLine += 1
      nextFrame = -1 if groundTruthBoxes[csvLine + 1] == '' else int(groundTruthBoxes[csvLine + 1].split(',')[-2])
      pointarr.append(np.array([[int(a), int(b)] for a,b in zip(groundTruthBoxes[csvLine].split(',')[0:8:2], groundTruthBoxes[csvLine].split(',')[1:8:2])]))
  frame_count += 1
  if skipFrames != 0: 
    skipFrames -= 1
    continue

  detections, borders = plate_detection(frame, get_localization_hyper_args(), get_recognition_hyper_args(), debug=True)
  # Add predicted box
  bbframe = frame.copy()
  for border in borders:
    bbframe = cv2.polylines(bbframe, [border], True, (255, 0, 0), 3)
  # Add ground truth box, new var for easier onclick event handling
  for points in pointarr:
    bbframe = cv2.polylines(bbframe, [points], True, (0, 255, 0), 3)

  # Display the original frame with bounding boxes
  cv2.imshow('Original frame', bbframe)
  
  # Display cropped plates
  for j, plate in enumerate(detections):
    cv2.imshow('Cropped plate #%d' % j, detections[j])

  # Display recognition results
  if get_args().stage == 1:
    print(segment_and_recognize(detections, get_recognition_hyper_args(), debug=True))

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
      if a == ord('b'):
        cropPlate = not cropPlate
        msg = "Recognition data collection mode" if cropPlate else "Localization data collection mode."
        cv2.setMouseCallback('Original frame', cropPlateEvent if cropPlate else captureBoxEvent)
        print(msg)
      if a == ord('p'):
        break
  # Press Q on keyboard to exit
  if a == ord('q'):
    break
  if a == ord('x'):
    skipFrames = 10
  if a == ord('c'):
    skipFrames = 50
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
