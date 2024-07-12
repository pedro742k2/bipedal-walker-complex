import time
import matplotlib.pyplot as plt
import os

avg_score_history = []
score_history = []


folder_name = "generated_plots"

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)


with open("logs/epochs.log", "r") as f:
    lines = f.readlines()

    score_list = []

    for line in lines:
        score_index = line.find("Epoch score:")
        avg_score_index = line.find("Average score:")
        # epoch_index = line.find("EPOCH #")

        # if epoch_index > 0:
        #     try:
        #         epoch = int(line[epoch_index+7:line.find("/")])
        #     except:
        #         pass

        if score_index > 0:
            score_history.append(float(line[score_index+13:-1]))

        if avg_score_index > 0:
            # print(line[score_index+33:-1])
            avg_score_history.append(float(line[score_index+33:-1]))

    x = [i for i in range(len(avg_score_history))]
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(x, avg_score_history)
    plt.scatter(x, score_history, color="orange")
    # plt.title("Score history")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend(["Scores average", "Scores"])
    plt.savefig(f"{folder_name}/scores_{time.time()}.png")

    plt.clf()

    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(x, avg_score_history)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(f"{folder_name}/avg_scores_{time.time()}.png")
