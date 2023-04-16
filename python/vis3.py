#! /usr/bn/env python3
#! -*- coding: utf-8 -*-

import numpy as np
import json
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import cv2



class Visualizer():
    def __init__(self, start, goal, occ_map):
        self.edges = set()
        self.final_path = None
        self.ci = np.inf

        self.sim = 0
        self.occ_map = cv2.cvtColor(occ_map, cv2.COLOR_BGR2RGB)


        self.all_new_edges, self.all_rem_edges, self.all_final_paths, self.all_cis = [], [], [], []
        self.lines = {}

        self.fig, self.ax = plt.subplots(figsize=(20, 20))

        self.all_final_edge_list = []


    def read_json(self, folder, max_iter=np.inf):
        files = sorted(os.listdir(folder))
        print(f"Found {len(files)} files")
        max_iter = min(max_iter, len(files))
        for i in range(max_iter):
            with open(os.path.join(folder, files[i]), "r") as f:
                data = json.load(f)

                self.all_new_edges.append(data['new_edges'])
                self.all_rem_edges.append(data['rem_edges'])
                self.all_final_paths.append(data['final_path'])
                self.all_cis.append(np.array(data['ci']))
                self.all_final_edge_list.append(data['final_edge_list'])

                print(f"Loaded {i+1}/{max_iter}", end="\r")



    def draw_final_path(self, fig, ax, path):
        if len(path) == 0:
            return
        path = np.array(path)
        x, y = path[:, 0], path[:, 1]
        ax.plot(
            y,
            x,
            color="green",
            lw=4,
            marker="x",
            markersize=4,
            markerfacecolor="blue",
            markeredgecolor="blue",
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


    def draw_ellipse(self, fig, ax, ci, start, goal, colour="b"):
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


    def draw_edge(self, fig, ax, edge):
        edge_tup = tuple(map(tuple, edge))

        
        l = ax.plot(
            [edge[0][1], edge[1][1]],
            [edge[0][0], edge[1][0]],
            color="red",
            lw=2,
            marker="x",
            markersize=4,
            markerfacecolor="blue",
            markeredgecolor="blue",
        )
        self.lines[edge_tup] = l

    def draw_tree(self, start, goal, sim):
        fig = self.fig
        ax = self.ax
        self.redraw_map(start, goal, self.occ_map, sim, fig, ax)
        
        print(f"Drawing Simulation {sim}")
        print(len(self.all_new_edges[sim]))

        for i in range(len(self.all_new_edges[sim])):
            new_edge = self.all_new_edges[sim][i]
            rem_edge = self.all_rem_edges[sim][i]
            path = self.all_final_paths[sim][i]
            ci = self.all_cis[sim][i]

            if rem_edge:
                for rem_e in rem_edge:
                    rem_e_tup = tuple(map(tuple, rem_e))
                    try:
                        ax.lines.remove(self.lines[rem_e_tup][0])
                        self.edges.remove(rem_e_tup)
                    except:
                        continue
                
                # fig, ax = self.redraw_map(start, goal, occ_map, sim, fig, ax)
                # for e in self.edges:
                #     self.draw_edge(fig, ax, e)

            if new_edge is not None:
                new_e_tup = tuple(map(tuple, new_edge))
                self.edges.add(new_e_tup)
                self.draw_edge(fig, ax, new_edge)

            if path is None:
                if self.final_path is not None:
                    self.draw_final_path(fig, ax, self.final_path)
            else:
                self.final_path = path
                self.draw_final_path(fig, ax, path)
            
            # print(f"Drawing CI {sim} - {ci}") 
            self.draw_ellipse(fig, ax, ci, start, goal, colour="b")
            ax.plot(start[0], start[1], "go", markersize=30)
            ax.plot(goal[0], goal[1], "ro", markersize=30)


            plt.show(block=False)
            plt.pause(0.0001)


    def draw_fast(self, start, goal, sim):
        fig, ax = self.fig, self.ax
        self.edges = self.all_final_edge_list[sim]
        path = self.all_final_paths[sim][-1]
        ci = self.all_cis[sim][0]

        self.redraw_map(start, goal, self.occ_map, sim, fig, ax)

        if path is not None:
            self.final_path = path
            self.draw_final_path(fig, ax, path)

        self.draw_ellipse(fig, ax, ci, start, goal, colour="b")
        ax.plot(start[0], start[1], "go", markersize=30)
        ax.plot(goal[0], goal[1], "ro", markersize=30)

        plt.show(block=False)
        plt.pause(1)


    def draw(self, start, goal, sim, fast):
        if fast:
            self.draw_fast(start, goal, sim)
        else:
            self.draw_tree(start, goal, sim)



    def redraw_map(self, start, goal, occ_map, sim, fig, ax):
        ax.cla()
        im = ax.imshow(occ_map, cmap=plt.cm.gray)

        for e in self.edges:
            self.draw_edge(fig, ax, e)

        ax.plot(start[0], start[1], "go", markersize=30)
        ax.plot(goal[0], goal[1], "ro", markersize=30)
        ax.set_title(f"BIT* - Simulation {sim}")
        ax.set_xlim(-10, 110)
        ax.set_ylim(-10, 110)
        

        return fig, ax


if __name__ == '__main__':
    log_dir = '../Logs/Default.png/'
    occ_map = np.array(Image.open('../gridmaps/Default.png'))
    start = np.array([0, 0])
    goal = np.array([99, 99])

    inv_map = np.where((occ_map==0)|(occ_map==1), occ_map^1, occ_map)
    visualizer = Visualizer(start, goal, occ_map = inv_map)

    print(log_dir, os.listdir(log_dir))
    
    visualizer.read_json(log_dir, max_iter=np.inf)

    for i in range(len(os.listdir(log_dir))):
        visualizer.draw(start, goal, i, True)
    