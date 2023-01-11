import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import Recognize

def cross_validate(file_path, hyper_args):
    plates = []
    names = []
    for f in os.listdir(file_path):
        names.append(f.split(".")[0])
        img = cv2.imread(file_path + "/" + f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :np.max(np.where(img != 0)[1] % img.shape[1]) + 1]
        plates.append(img)

    results = Recognize.segment_and_recognize(plates, hyper_args)

    print("Ground Truth:\n", names)
    print("Results:\n", results)
