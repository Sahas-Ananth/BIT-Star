#! /usr/bn/env python3
#! -*- coding: utf-8 -*-

import numpy as np
import json
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import os
import time


def read_json(folder, max_iter=np.inf):
    all_edges, all_cis, all_paths = [], [], []
    fold_path = f"{os.path.abspath(os.path.dirname(__file__))}/../Logs/PyViz/{folder}/"

    files = sorted(os.listdir(fold_path))
    print(f"Found {len(files)} files")
    max_iter = min(max_iter, len(files))
    for i in range(max_iter):
        edges, cis, paths = [], [], []
        with open(fold_path + files[i], "r") as f:
            data = json.load(f)
            for edge in data["edges"]:
                if len(edge) == 0:
                    continue
                # for j in range(len(edge)):
                #     # edges.append((edge[j][0], edge[j][1]))
                #     edges.append(edge[j])
                edges.append(edge)

            cis.append(data["ci"])

            for path in data["final_path"]:
                if path is None:
                    continue
                paths.append(path)
            print(f"Loaded {i+1}/{max_iter}", end="\r")

        # all_edges.append(np.array(edges))
        all_edges.append(edges)
        all_cis.append(np.array(cis).squeeze())
        all_paths.append(paths)

    return all_edges, all_cis, all_paths


def draw_final_path(fig, ax, path):
    if len(path) == 0:
        return
    print(f"Plotting Final Path")
    path = np.array(path)
    x, y = path[:, 0], path[:, 1]
    ax.plot(
        y,
        x,
        color="green",
        lw=6,
        marker="o",
        markersize=4,
        markerfacecolor="green",
        markeredgecolor="green",
    )
    for i in range(len(path) - 1):
        ax.text(
            y[i],
            x[i] + 0.5,
            f"({y[i]:.5f}, {x[i]:.5f})",
            color="black",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
        )


def draw_ellipse(fig, ax, ci, start, goal, colour="b"):
    if ci == np.inf:
        return
    cmin = np.linalg.norm(goal - start)
    center = (start + goal) / 2.0
    r1 = ci
    r2 = np.sqrt(ci**2 - cmin**2)
    theta = np.arctan2(goal[1] - start[1], goal[0] - start[0])
    theta = np.degrees(theta)
    patch = Ellipse(center, r1, r2, theta, color=colour, fill=False, lw=5, ls="--")
    ax.add_patch(patch)


def draw_edges(fig, ax, edges):
    if len(edges) == 0:
        return
    # x1, y1, x2, y2 = edges[:, 0, 0], edges[:, 0, 1], edges[:, 1, 0], edges[:, 1, 1]
    s_time = time.time()
    for e in edges:
        x1, y1, x2, y2 = e[0][0], e[0][1], e[1][0], e[1][1]
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
    print(f"Draw edges: {time.time() - s_time}")


def draw_tree(sim_edges, sim_costs, sim_path, start, goal):
    for sim in range(len(sim_edges)):
        fig, ax = plt.subplots(figsize=(20, 20))

        ax.add_patch(
            plt.Rectangle(
                (30, 0),
                20,
                30,
                fill=True,
                color="black",
                linewidth=2,
            )
        )

        ax.add_patch(
            plt.Rectangle(
                (30, 31),
                20,
                30,
                fill=True,
                color="black",
                linewidth=2,
            )
        )

        edges = sim_edges[sim]
        draw_final_path(fig, ax, sim_path[sim][-1])
        print(f"Drawing Simulation {sim}")
        draw_edges(fig, ax, edges[-1])
        print(f"Drawing CI {sim} - {sim_costs[sim]}")
        draw_ellipse(fig, ax, max(sim_costs[sim]), start, goal, colour="b")
        # draw_ellipse(fig, ax, min(sim_costs[sim]), start, goal, colour="k")
        ax.plot(start[0], start[1], "go", markersize=10)
        ax.plot(goal[0], goal[1], "ro", markersize=10)
        ax.set_title(f"BIT* - Simulation {sim}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        plt.show()


def main():
    folder = "2023-04-14 20:20:25.757163"
    sim_edges, sim_costs, sim_paths = read_json(folder, max_iter=np.inf)

    start = np.array([0, 0])
    goal = np.array([99, 0])
    # self.map = np.ones(size)
    # unique_sim_costs = sorted(np.unique(np.concatenate(sim_costs)), reverse=True)
    # unique_sim_paths = []
    # for paths in sim_path:
    #     unique_sim_paths.append(np.unique(paths, axis=1))

    # print(unique_sim_paths)
    # exit()
    # print(type(unique_sim_paths[0]))
    # print(sim_costs)
    # print(sim_paths)
    # exit()
    draw_tree(sim_edges, sim_costs, sim_paths, start, goal)
    plt.show()


if __name__ == "__main__":
    main()
