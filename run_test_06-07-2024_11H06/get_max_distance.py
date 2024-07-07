import numpy as np

avg_score_history = []
score_history = []

biggest_distance = float("+inf")
biggest_distance_coords = [0, 0]


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


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
            if x < 0 or y < 0:
                continue

            current_distance = euclidean_distance([x, y], TARGET_POSITION)

            if current_distance < biggest_distance:
                biggest_distance = current_distance
                biggest_distance_coords = [x, y]

    print(biggest_distance, biggest_distance_coords)

    # x = [i for i in range(len(avg_score_history))]
    # plt.figure(figsize=(12, 8), dpi=300)
    # plt.plot(x, avg_score_history)
    # plt.scatter(x, score_history, color="orange")
    # # plt.title("Score history")
    # plt.xlabel("Epoch")
    # plt.ylabel("Score")
    # plt.legend(["Scores average", "Scores"])
    # plt.savefig(f"scores_{time.time()}.png")
