import time
from matplotlib import pyplot as plt
import numpy as np

coords = []

# def euclidean_distance(point1, point2):
#     return np.linalg.norm(np.array(point1) - np.array(point2))


TARGET_POSITION = [20, 20]

with open("logs/epochs.log", "r") as f:
    lines = f.readlines()

    score_list = []

    for line in lines:
        distance_index = line.find("- Reached coords:")

        if distance_index > 0:
            target_line = line[distance_index+18:-1]
            target_line = target_line.removeprefix("[")
            target_line = target_line.removesuffix("]")
            target_line = target_line.split(",")
            x = float(target_line[0])
            y = float(target_line[1])
            coords.append([x, y])

    x = [x for x, _ in coords]
    y = [y for _, y in coords]
    plt.figure(figsize=(12, 8), dpi=300)
    plt.xlim(left=-30, right=30)
    plt.ylim(bottom=-30, top=30)

    plt.hlines([0, 10, 20], xmin=[-30, -30, -30],
               xmax=[0, 10, 20], linestyles="dashed", colors="black")
    plt.vlines([0, 10, 20], ymin=[-30, -30, -30], ymax=[0, 10, 20],
               linestyles="dashed", colors="black")

    plt.scatter(x, y, color="blue")
    plt.scatter(20, 20, color="red")
    plt.scatter(0, 0, color="black")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["", "", "Reached coords", "Target coords", "origin"])
    plt.savefig(f"distances_{time.time()}.png")
