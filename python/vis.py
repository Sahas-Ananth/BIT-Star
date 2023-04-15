#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import numpy as np
import cv2
import json
import os
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp
from copy import deepcopy


def get_data_from_json(folder, max_iter=np.inf):
    all_edges, all_cis, all_paths = [], [], []
    fold_path = f"{os.path.abspath(os.path.dirname(__file__))}/../Logs/PyViz/{folder}/"

    files = sorted(os.listdir(fold_path))
    max_iter = min(max_iter, len(files) - 1)
    for i in range(max_iter):
        edges, cis, paths = [], [], []
        with open(fold_path + files[i], "r") as f:
            data = json.load(f)
            for edge in data["edges"]:
                if len(edge) == 0:
                    continue
                for j in range(len(edge)):
                    edges.append((edge[j][0], edge[j][1]))

            cis.append(data["ci"])

            for path in data["final_path"]:
                if path is None:
                    continue
                for j in range(len(path)):
                    paths.append(tuple(path[j]))
            print(f"Loaded {i+1}/{max_iter}", end="\r")

        all_edges.append(np.array(edges))
        all_cis.append(np.array(cis).squeeze())
        all_paths.append(np.array(paths))

    return all_edges, all_cis, all_paths


def draw_ellipse(fig, ax, ci):
    if ci == np.inf:
        return
    cmin = np.linalg.norm(goal_arr - start_arr)
    center = (start_arr + goal_arr) / 2.0
    r1 = ci
    r2 = np.sqrt(ci**2 - cmin**2)
    theta = np.arctan2(goal[1] - start[1], goal[0] - start[0])
    theta = np.degrees(theta)
    ax.add_patch(
        Ellipse(center, r1, r2, theta, color="blue", fill=False, lw=5, ls="--")
    )


def grow_edges(num, x0, y0, x1, y1, line):
    line.set_data([y0[:num], y1[:num]], [x0[:num], x1[:num]])
    return line


def draw_edges(fig, ax, edges):
    x1, y1, x2, y2 = edges[:, 0, 0], edges[:, 0, 1], edges[:, 1, 0], edges[:, 1, 1]
    ax.plot(
        [y1, y2],
        [x1, x2],
        color="red",
        lw=2,
        marker="x",
        markersize=4,
        markerfacecolor="blue",
        markeredgecolor="blue",
    )

    print("Finished animating edges")
    plt.show()
    return None


def draw_tree(it, copy_image):
    print(f"Process started for iteration {it} at {mp.current_process().name}")
    img_folder = (
        f"{os.path.abspath(os.path.dirname(__file__))}/../Output/PyViz/{experiment}/"
    )
    fig, ax = plt.subplots(figsize=(20, 20))
    # plt.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))

    # plt.axis("off")
    ax.set_title(f"Iteration {it}", fontsize=30)

    draw_ellipse(fig, ax, unique_cis[it][1])
    print("len Edges: ", len(e))
    # ani = draw_edges(fig, ax, e[it])
    draw_edges(fig, ax, e[it])

    ax.plot(start[0], start[1], color="green", marker="o", markersize=10)
    ax.plot(goal[0], goal[1], color="orange", marker="o", markersize=10)

    # os.makedirs(
    #     img_folder,
    #     exist_ok=True,
    # )
    # # writervideo = animation.FFMpegWriter(fps=30)
    # # ani.save(f"{img_folder}iter_{it:02d}.mp4", writer=writervideo)
    # print("Saved gif")
    # plt.savefig(
    #     f"{img_folder}iter_{it:02d}.png",
    #     bbox_inches="tight",
    #     pad_inches=0.4,
    # )
    plt.axis("off")

    plt.close()
    print(f"Process finished for iteration {it} at {mp.current_process().name}")
    return f"{img_folder}iter_{it:02d}.png"


experiment = "2023-04-11 17:56:30.389629"
experiment = "2023-04-12 17:12:34.226086"
experiment = "2023-04-12 22:41:54.723595"
experiment = "2023-04-13 15:12:55.266714"
experiment = "2023-04-13 22:45:15.197294"
max_iter = np.inf
e, c, p = get_data_from_json(experiment, max_iter=max_iter)
unique_edges = []
unique_cis = []
unique_paths = []
for i in range(len(c)):
    unique_e_indices = np.unique(e[i], axis=0, return_index=True)[1]
    unique_e = [e[i][j] for j in sorted(unique_e_indices)]

    # unique_edges.append(np.unique(e[i], axis=0))
    unique_edges.append(np.array(unique_e))
    unique_cis.append(np.unique(c[i]))
    unique_paths.append(np.unique(p[i], axis=0))


print("Unique edges: ", len(unique_edges[i]), unique_edges[i].shape)
print("Unique cis: ", len(unique_cis[i]))
print("Unique paths: ", len(unique_paths[i]))

# image = cv2.imread(
#     f"{os.path.abspath(os.path.dirname(__file__))}/../gridmaps/occupancy_map.png"
# )
unique_e_indices
image = np.ones((100, 100, 3), dtype=np.uint8) * 255

copy_image = image.copy()

start = (0, 0)
goal = (99, 99)
start_arr = np.array(start)
goal_arr = np.array(goal)

print("\n\nPlotting...")

# processes = []
# output = mp.Queue()

for i in range(len(unique_edges)):
    draw_tree(i, image.copy())

# for i in range(len(unique_edges[:max_iter])):
#     process = mp.Process(target=draw_tree, args=(0, image.copy()))
# processes.append(process)
# process.start()

# for process in processes:
#     process.join()
#     print(f"Process {process.name} joined")


# TODO: Fix threading issue with matplotlib https://stackoverflow.com/questions/31719138/matplotlib-cant-render-multiple-contour-plots-on-django
