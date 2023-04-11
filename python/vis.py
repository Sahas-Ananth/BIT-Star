#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import numpy as np
import cv2
import json
import os
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
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
            print(f"Loaded {i+1}/{len(files) - 1}", end="\r")

        all_edges.append(np.array(edges))
        all_cis.append(np.array(cis).squeeze())
        all_paths.append(np.array(paths))

    return all_edges, all_cis, all_paths


def draw_ellipse(ci):
    if ci == np.inf:
        return
    cmin = np.linalg.norm(goal_arr - start_arr)
    center = (start_arr + goal_arr) / 2.0
    r1 = ci
    r2 = np.sqrt(ci**2 - cmin**2)
    theta = np.arctan2(goal[1] - start[1], goal[0] - start[0])
    theta = np.degrees(theta)
    plt.gca().add_patch(
        Ellipse(center, r1, r2, theta, color="blue", fill=False, lw=5, ls="--")
    )


def grow_edges():
    pass


def draw_edges(edges):
    x1, y1, x2, y2 = edges[:, 0, 0], edges[:, 0, 1], edges[:, 1, 0], edges[:, 1, 1]
    plt.plot(
        [y1, y2],
        [x1, x2],
        color="red",
        linewidth=1,
        marker="x",
        markersize=4,
        markerfacecolor="blue",
        markeredgecolor="blue",
    )


def draw_tree(it, copy_image):
    print(f"Process started for iteration {it} at {mp.current_process().name}")
    img_folder = (
        f"{os.path.abspath(os.path.dirname(__file__))}/../Output/PyViz/{experiment}/"
    )
    plt.figure(figsize=(20, 20))
    # plt.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))

    plt.axis("off")
    plt.title(f"Iteration {it}", fontsize=30)

    draw_ellipse(unique_cis[it][1])
    draw_edges(e[it])

    plt.plot(start[0], start[1], color="green", marker="o", markersize=10)
    plt.plot(goal[0], goal[1], color="orange", marker="o", markersize=10)

    os.makedirs(
        img_folder,
        exist_ok=True,
    )
    plt.savefig(
        f"{img_folder}iter_{it:02d}.png",
        bbox_inches="tight",
        pad_inches=0.4,
    )
    plt.axis("off")
    plt.close()
    return f"{img_folder}iter_{it:02d}.png"


experiment = "2023-04-11 16:55:45.976899"
e, c, p = get_data_from_json(experiment)
unique_edges = []
unique_cis = []
unique_paths = []
for i in range(len(c)):
    unique_edges.append(np.unique(e[i], axis=0))
    unique_cis.append(np.unique(c[i]))
    unique_paths.append(np.unique(p[i], axis=0))

print("Unique edges: ", len(unique_edges[i]))
print("Unique cis: ", len(unique_cis[i]))
print("Unique paths: ", len(unique_paths[i]))

# image = cv2.imread(
#     f"{os.path.abspath(os.path.dirname(__file__))}/../gridmaps/occupancy_map.png"
# )

image = np.ones((100, 100, 3), dtype=np.uint8) * 255

copy_image = image.copy()

start = (0, 0)
goal = (99, 99)
start_arr = np.array(start)
goal_arr = np.array(goal)

print("\n\nPlotting...")

processes = []
output = mp.Queue()
for i in range(len(e)):
    process = mp.Process(target=draw_tree, args=(i, image.copy()))
    processes.append(process)
    process.start()

for process in processes:
    print(f"Process {process.name} joined")
    process.join()


# TODO: Fix threading issue with matplotlib https://stackoverflow.com/questions/31719138/matplotlib-cant-render-multiple-contour-plots-on-django
