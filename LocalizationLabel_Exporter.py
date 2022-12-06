import csv
import pandas as pd
import re

header = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'frame', 'time']

# all data is contained within "labeling.txt"
# file name is swapped whenever we wanna export a new test set

data = list()
for row in open('./BoundingBoxGroundTruth.txt', 'r').read().split("\n"):
    if '-' in row: continue
    l = list(map(lambda s: int(s), filter(lambda s: re.search("\\d+", s), re.split("\\D+", row))))
    l = list([l[2], l[3], l[6], l[7], l[4], l[5], l[0], l[1], l[8], l[8] / 12.0])
    data.append(tuple(l))
pd.DataFrame(data, columns=header).to_csv("BoundingBoxGroundTruth.csv", index=False)
