import numpy as np
from sklearn.model_selection import train_test_split
import Localization
import cv2
import os
import argparse
from itertools import product

cwd = os.path.abspath(os.getcwd())

def cross_validate(file_path, hyper_args, rec_hyper_args):
    images = []
    groundTruthBoxes = open("BoundingBoxGroundTruth.csv", "r").read().split('\n')
    boundingBoxes = []
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened() == False: print("Error opening video stream or file")
    csvLine = 0
    nextFrame = -1 if groundTruthBoxes[csvLine + 1] == '' else int(groundTruthBoxes[csvLine + 1].split(',')[-2])
    frame_count = 0
    last_ground_truth = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        currFrameBoundingBoxes = []
        while nextFrame == frame_count:
            csvLine += 1
            currFrameBoundingBoxes.append([(int(a), int(b)) for a, b in zip(groundTruthBoxes[csvLine].split(',')[0:8:2],
                                                                     groundTruthBoxes[csvLine].split(',')[1:8:2])])
            nextFrame = -1 if groundTruthBoxes[csvLine + 1] == '' else int(groundTruthBoxes[csvLine + 1].split(',')[-2])
        if len(currFrameBoundingBoxes) == 0: 
            currFrameBoundingBoxes = last_ground_truth
        else:
            currFrameBoundingBoxes = sorted(currFrameBoundingBoxes, key=lambda b: b[0][0])
            last_ground_truth = currFrameBoundingBoxes
        images.append(frame)
        boundingBoxes.append(currFrameBoundingBoxes)
        frame_count += 1
        
    cap.release()
    cv2.destroyAllWindows()
    train_and_test_model(images, boundingBoxes, hyper_args, rec_hyper_args)

def train_and_test_model(data, labels, hyper_args, rec_hyper_args):
    best_hyper_arg = None
    best_train = 0

    data = data[1731:]
    labels = labels[1731:]
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)
    x_train = x_train[:50]
    y_train = y_train[:50]
    runs = 0
    for v in product(*hyper_args.values()):
        print(runs)
        runs += 1
        hyper_arg_dict = dict(zip(hyper_args, v))
        parser = argparse.ArgumentParser()
        for k, v in hyper_arg_dict.items():
            parser.add_argument('--' + str(k), type=type(v), default=v)
        hyper_arg = parser.parse_args()
        _, overlap = evaluate_bounding_boxes(x_train, y_train, hyper_arg, rec_hyper_args)
        if overlap > best_train or best_train == 0: #natural selection of results that improve
            best_train = overlap
            best_hyper_arg = hyper_arg

    best_test = evaluate_bounding_boxes(x_test, y_test, best_hyper_arg, rec_hyper_args)

    print("Best match: ")
    print("Train set: " + str(best_train))
    print("Test set: " + str(best_test))
    print("Best hyper-parameters: " + str(best_hyper_arg))
    return best_hyper_arg

# Using regular shoelace area formula for any polygon possible
# NB! WE ASSUME POLYGONS ARE ALWAYS CONVEX, HENCE FOR PROPER CALCULATION
# MAKE SURE THE COORDINATES FOLLOW A (COUNTER-)CLOCKWISE ORDER
def shoelaceArea(box):
    x, y = zip(*box)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Use Jordan curve's theorem for a ray-casting algorithm
def isContained(p, b, offset = 0): 
    # Boundary case
    if p in set(b): return True
    # Pick arbitrary ray for testing intersections
    testline = [p, (1000, p[1] + offset)]
    c = 0
    intersections = set()
    # For each side of the box, check if the ray intersects the side
    for i in range(0, 4):
        checkline = [b[i], b[(i + 1) % 4]]
        intersect = lineIntersect(testline[0], testline[1], checkline[0], checkline[1]) 
        if intersect is not None:
            # If intersects two sides on same point - intersects edgepoint
            # Offset ray and redo containment check
            if intersect in intersections: return isContained(p, b, offset + 10)
            else:
                intersections.add(intersect)
                c += 1
    # If intersects even number of lines, outside the polygon, otherwise in
    return c % 2

def lineIntersect(a, b, c, d): 
    # Build first line from a and b
    a1 = b[1] - a[1]
    b1 = a[0] - b[0]
    c1 = a1 * a[0] + b1 * a[1]
    # Build second line from c and d
    a2 = d[1] - c[1]
    b2 = c[0] - d[0]
    c2 = a2 * c[0] + b2 * c[1]

    det = a1 * b2 - a2 * b1
    # If det == 0, then lines are parallel
    if det == 0: return None
    # Potential intersection, x coord
    potx = (b2 * c1 - b1 * c2) / det
    # If not within both segments, return None
    if potx > max(a[0], b[0]) or potx > max(c[0], d[0]) or potx < min(a[0], b[0]) or potx < min(c[0], d[0]): return None
    # Potential intersection, y coord
    poty = (a1 * c2 - a2 * c1) / det
    # If not within both segments, return None
    if poty > max(a[1], b[1]) or poty > max(c[1], d[1]) or poty < min(a[1], b[1]) or poty < min(c[1], d[1]): return None 
    # Turn -0 to 0, else unchanged
    return (0 if potx == -0 else potx, 0 if poty == -0 else poty)

def intersect(box1, box2):
    # Build overlapping quad endpoints
    coords = set()
    # Check, for every point, whether it lies inside the other polygon
    for p in box1: 
        if isContained(p, box2): coords.add(p)
    for p in box2:
        if isContained(p, box1): coords.add(p)
    # Check intersection of every pair of polygon segments
    for i in range(0, 4):
        for j in range(0, 4):
            p1 = box1[i]
            p2 = box1[(i + 1) % 4]
            p3 = box2[j]
            p4 = box2[(j + 1) % 4]
            li = lineIntersect(p1, p2, p3, p4)
            if li is not None: 
                coords.add(li)
    # If not enough intersection points, return a point (no intersection)
    if len(coords) < 4: return [(0, 0), (0, 0), (0, 0), (0, 0)]
    # Sort coordinates in counter-clockwise order for proper shoelace area
    # Get some middle point (average of all, assumes polygon is convex)
    # Find angle between that point and all edge points
    avgx = np.average(list(map(lambda p: p[0], list(coords))))
    avgy = np.average(list(map(lambda p: p[1], list(coords))))
    return sorted(list(coords), key=lambda p: np.arctan2(p[1] - avgy, p[0] - avgx))


def evaluate_single_box(model_box, test_box, img=None, i=0):
    if set(test_box) == {(0, 0), (0, 0), (0, 0), (0, 0)}: #in the unlikely case the default has a match
        if set(model_box) == {(0, 0), (0, 0), (0, 0), (0, 0)}: return 1, 1
        else: return 0, 0

    area_model_box = shoelaceArea(model_box)
    area_test_box = shoelaceArea(test_box)

    intersection = intersect(model_box, test_box)
    area_intersection = shoelaceArea(intersection)
    area_union = area_model_box + area_test_box - area_intersection
    
    overlap = area_intersection / area_union

    if img is not None: #visual debug to append intersection percentages for different images directly - not implementation important
        global cwd
        if not os.path.exists(os.path.join(cwd, "images")):
            os.makedirs(os.path.join(cwd, "images"))
        show_img = img.copy()
        show_img = cv2.polylines(show_img, [np.array([list(ele) for ele in model_box])], True, (255, 0, 0), 3)
        show_img = cv2.polylines(show_img, [np.array([list(ele) for ele in test_box])], True, (0, 255, 0), 3)
        show_img = cv2.putText(show_img, str(overlap), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(cwd, "images", "frame%d.jpg" % i), show_img)

    success = 1 if overlap > 0.75 else 0
    return success, overlap


def evaluate_bounding_boxes(x, y, hyper_args, rec_hyper_args):
    boxes = []
    default = [(0, 0), (0, 0), (0, 0), (0, 0)]

    hyper_args.memoize_bounding_boxes = False
    for i in range(0, len(x)):
        res = Localization.plate_detection(x[i], hyper_args, rec_hyper_args)[1]
        # If resulting detections are more than ground truths -> get only best guesses
        if len(res) > len(y[i]): 
            res = sorted(res, key=lambda b: np.max([evaluate_single_box(list(map(tuple, b)), yb) for yb in y[i]]), reverse=True)[:len(y[i])]
        # If resulting detections are less than ground truths -> add defaults
        while len(res) != len(y[i]): res.append(default)
        boxes.append(res)

    successScore = 0
    overlapScore = 0
    total = 0

    # Frameboxes refers to many bounding boxes on same frame (for Category 3, for ex)
    for i in range(0, len(boxes)): 
        # Sort by top-left vertex, x coordinate (when multiple plates on same frame)
        frameboxes = sorted(boxes[i], key=lambda b: b[0][0])
        for j in range(0, len(frameboxes)):
            fb = list(map(tuple, frameboxes[j]))
            # ss - success score, os - overlap score
            ss, os = evaluate_single_box(fb, y[i][j], x[i], i)
            #print('--------------')
            #print(fb)
            #print(y[i][j])
            #print(ss)
            #print(os)
            successScore += ss
            overlapScore += os
            total += 1
  
    successScore /= total
    overlapScore /= total

    #print("Hyper parameters:" + str(hyper_args))
    #print("Score (successful matches):" + str(successScore * 100.0) + "%")
    #print("Score (total overlap):" + str(overlapScore * 100.0) + "%")

    return (successScore * 100.0, overlapScore * 100.0)