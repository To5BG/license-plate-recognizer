import numpy as np
from sklearn.model_selection import train_test_split
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
            images.append(frame)
            nextFrame = -1 if groundTruthBoxes[csvLine + 1] == '' else int(groundTruthBoxes[csvLine + 1].split(',')[-2])
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(labels)

    train_model(images[:40], labels[:40], hyper_args, sizes)


def evaluate_single_box(model_box, test_box, index):
    area_model_box = (model_box[1][0] - model_box[0][0]) * (model_box[3][1] - model_box[0][1])
    area_test_box = (test_box[1][0] - test_box[0][0]) * (test_box[3][1] - test_box[0][1])

    # consider whether you need to invert the np.min and np.max for the fact that y is inverted
    area_intersection = max(0, min(model_box[1][0], test_box[1][0]) - max(model_box[0][0], test_box[0][0])) * \
                        max(0, min(model_box[3][1], test_box[3][1]) - max(model_box[0][1], test_box[0][1]))
    #print("Intersection: " + str(area_intersection))

    area_union = area_model_box + area_test_box - area_intersection

    overlap = area_intersection / area_union
    #print(model_box)
    # if overlap < 0.6:
    #     print(model_box)
    #     print(test_box)
    #     print(overlap)
    #     print(index)
    #     print()


    success = 1 if overlap > 0.75 else 0

    return success


def evaluate_bounding_boxes(x, y, hyper_args, size):
    boxes = []
    y = np.array(y)
    default = np.zeros(y[0].shape)

    for img in x:
        a = Localization.plate_detection(img, hyper_args)[1]
        boxes.append(a[0] if len(a) > 0 else default)
        default = a[0] if len(a) > 0 else default

    # converts to the same shape cuz for some god forsaken reason it is not
    boxes = np.squeeze(np.array(boxes))
    score = 0
    for i in range(len(boxes)): score += evaluate_single_box(boxes[i], y[i], i+1) / len(boxes)

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
