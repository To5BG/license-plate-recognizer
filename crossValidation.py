import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.path as mlp
import Localization
import cv2

# define a get_sizes
def cross_validation(file_path, hyper_args, sizes=[0.1]):
    images = []
    groundTruthBoxes = open("BoundingBoxGroundTruth.csv", "r").read().split('\n')
    labels = []
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened() == False: print("Error opening video stream or file")
    csvLine = 0
    nextFrame = -1 if groundTruthBoxes[csvLine + 1] == '' else int(groundTruthBoxes[csvLine + 1].split(',')[-2])
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        while nextFrame == frame_count:
            csvLine += 1
            labels.append(np.array([[int(a), int(b)] for a, b in zip(groundTruthBoxes[csvLine].split(',')[0:8:2],
                                                                     groundTruthBoxes[csvLine].split(',')[1:8:2])]))
            print(labels)
            images.append(frame)
            nextFrame = -1 if groundTruthBoxes[csvLine + 1] == '' else int(groundTruthBoxes[csvLine + 1].split(',')[-2])
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    train_model(images, labels, hyper_args, sizes)

def shoelaceArea(box):
    x, y = zip(*box)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def intersect(box1, box2):
    # Build overlapping quad endpoints and apply shoelaceArea formula
    coords = list()
    def isContained(p, b): mlp.Path(np.array(b)).contains_point(p)
    def helper(a, b, c): return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    def lineIntersect(a, b, c, d): return helper(a, c, d) != helper(b, c, d) and helper(a, b, c) != helper(a, b, d)
    for p in box1: 
        if isContained(p, box2): coords.append(p)
    for p in box2:
        if isContained(p, box1): coords.append(p)
    for i in range(4):
        for j in range(i, 4):
            p1 = box1[i]
            p2 = box1[(i + 1) % 4]
            p3 = box2[j]
            p4 = box2[(j + 1) % 4]
            if lineIntersect(p1, p2, p3, p4): 
                denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
                u = (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0]) / denom
                coords.append([p1[0] + u * (p2[0] - p1[0]), p1[1] + u * (p2[1] - p1[1])])


def evaluate_single_box(model_box, test_box):
    print(model_box)
    print(test_box)
    print('---------')
    area_model_box = shoelaceArea(model_box)
    area_test_box = shoelaceArea(test_box)

    # consider whether you need to invert the np.min and np.max for the fact that y is inverted
    area_intersection = shoelaceArea(intersect(model_box, test_box))
    #print("Intersection: " + str(area_intersection))
    area_union = area_model_box + area_test_box - area_intersection

    overlap = area_intersection / area_union
    success = 1 if overlap > 0.75 else 0
    return success


def evaluate_bounding_boxes(x, y, hyper_args, size):
    boxes = []
    y = np.array(y)
    default = np.zeros(y[0].shape)

    # for img in x_train:
    #     a = Localization.plate_detection(img, hyper_args)[1]
    #     plates_train.append(a[0] if len(a) > 0 else default)
    #     default = a[0] if len(a) > 0 else default
    # for img in x_test:
    #     a = Localization.plate_detection(img, hyper_args)[1]
    #     plates_test.append(a[0] if len(a) > 0 else default)
    #     default = a[0] if len(a) > 0 else default

    # # converts to the same shape cuz for some god forsaken reason it is not
    # plates_train = np.squeeze(np.array(plates_train))
    # plates_test = np.squeeze(np.array(plates_test))

    score_train = 0
    score_test = 0
    for i in range(len(plates_train)): score_train += evaluate_single_box(plates_train[i], y_train[i], i)
    for i in range(len(plates_test)): score_test += evaluate_single_box(plates_test[i], y_test[i], i)
    score_train /= (len(plates_train))
    score_test /= (len(plates_test))

    print("Hyper parameters:" + str(hyper_args))
    print("Size:" + str(size))
    print("Score:" + str(score * 100.0) + "%")

    return score * 100.0


def train_model(data, labels, hyper_args, sizes):
    best = 0
    best_hyper_arg = []
    best_size = []
    hyper_args = np.array([hyper_args]) #hardcoded solution for now, hyperparameters should be an array in the future
    test_x = []
    test_y = []


    for hyper_arg in hyper_args:
        for size in sizes:
            x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=size, random_state=42,
                                                                shuffle=False)
            res = evaluate_bounding_boxes(x_train, y_train, hyper_arg, size)
            if res > best: #natural selection of results that improve
                best = res
                best_hyper_arg.append(hyper_arg)
                best_size.append(size)
                test_x.append(x_test)
                test_y.append(y_test)

    best = 0
    bHp = None
    bS = None
    print(best_hyper_arg)
    for i in range(len(best_hyper_arg)):
        res = evaluate_bounding_boxes(test_x.pop(), test_y.pop(), best_hyper_arg[i], best_size[i])
        if res > best: #natural selection of results that improve
            best = res
            bHp = best_hyper_arg[i]
            bS = best_size[i]

    print("Best match: " + str(best) + "%\n hyper_arg = " + str(bHp) + "\n size = " + str(bS))

    return best_hyper_arg, best_size
