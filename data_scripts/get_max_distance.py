import numpy as np

closest_distance = float("+inf")
highest_x_distance = float("-inf")
highest_y_distance = float("-inf")
closest_distance_coords = []

closest_distance_epoch = 0
highest_x_epoch = 0
highest_y_epoch = 0
epochs = 0


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


TARGET_POSITION = [20, 20]

with open("logs/epochs.log", "r") as f:
    lines = f.readlines()

    score_list = []

    for line in lines:
        distance_index = line.find("- Reached coords:")

        if distance_index > 0:
            epochs += 1

            target_line = line[distance_index+18:-1]
            target_line = target_line.removeprefix("[")
            target_line = target_line.removesuffix("]")
            target_line = target_line.split(",")
            x = float(target_line[0])
            y = float(target_line[1])

            if x > highest_x_distance:
                highest_x_epoch = epochs
                highest_x_distance = x

            if y > highest_y_distance:
                highest_y_epoch = epochs
                highest_y_distance = y

            current_distance = euclidean_distance([x, y], TARGET_POSITION)

            if current_distance < closest_distance:
                closest_distance_epoch = epochs
                closest_distance = current_distance
                closest_distance_coords = [x, y]

    print(f"Info ({epochs} epochs):")
    print(
        f"\t - Closest distance from target (epoch #{closest_distance_epoch}): {closest_distance}")
    print(
        f"\t - Closest coords from target (epoch #{closest_distance_epoch}): {closest_distance_coords}")
    print(
        f"\t - Highest x coord (epoch #{highest_x_epoch}): {highest_x_distance}")
    print(
        f"\t - Highest y coord (epoch #{highest_y_epoch}): {highest_y_distance}")
