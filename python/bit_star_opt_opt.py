#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from queue import PriorityQueue
import numpy as np
from PIL import Image
import random
import cProfile, pstats
import time
from datetime import datetime
import os
import json

random.seed(1)
np.random.seed(1)

global start_arr
global goal_arr


class Node:
    def __init__(self, coords, parent=None, gt=np.inf, par_cost=None):
        self.x = coords[0]
        self.y = coords[1]
        self.tup = (self.x, self.y)
        self.np_arr = np.array([self.x, self.y])

        self.parent = parent
        self.par_cost = par_cost
        self.gt = gt

        global start_arr
        self.start = start_arr
        global goal_arr
        self.goal = goal_arr

        self.g_hat = self.gen_g_hat()
        self.h_hat = self.gen_h_hat()
        self.f_hat = self.g_hat + self.h_hat

        self.children = set()

    def gen_g_hat(self):
        return np.linalg.norm(self.np_arr - self.start)

    def gen_h_hat(self):
        return np.linalg.norm(self.np_arr - self.goal)

    def __str__(self) -> str:
        return str(self.tup)

    def __repr__(self) -> str:
        return str(self.tup)


class Map:
    def __init__(self, start, goal, image_path=None, size=(5, 5)):
        self.start = start
        self.goal = goal
        self.obstacles = set()
        self.dim = 2
        if image_path is None:
            self.map = np.ones(size)
            self.map[0:30, 30:50] = 0
            self.map[31:61, 30:50] = 0
        else:
            self.map = np.array(Image.open(image_path))
        # self.map = np.ones((5, 5))
        ind = np.argwhere(self.map > 0)
        self.free = set(list(map(lambda x: tuple(x), ind)))
        ind = np.argwhere(self.map == 0)
        self.occupied = set(list(map(lambda x: tuple(x), ind)))
        self.get_f_hat_map()

    def sample(self):
        while True:
            x, y = np.random.uniform(0, self.map.shape[0]), np.random.uniform(
                0, self.map.shape[1]
            )
            if (int(x), int(y)) in self.free:
                return (x, y)

    def new_sample(self):
        while True:
            free_node = random.sample(self.free, 1)[0]
            noise = np.random.uniform(0, 1, self.dim)
            new_node = free_node + noise
            if (int(new_node[0]), int(new_node[1])) in self.free:
                return new_node

    def get_f_hat_map(self):
        global start_arr, goal_arr
        map_x, map_y = self.map.shape
        self.f_hat_map = np.zeros((map_x, map_y))
        for x in range(map_x):
            for y in range(map_y):
                #! Potential BUG: Possible bug here with the Node class not having gt, parent, par_cost initialized.
                f_hat = np.linalg.norm(
                    np.array([x, y]) - np.array(goal_arr)
                ) + np.linalg.norm(np.array([x, y]) - np.array(start_arr))
                self.f_hat_map[x, y] = f_hat


class bitstar:
    def __init__(self, start, goal, occ_map, no_samples=20, rbit=100, dim=2):
        self.start = start
        self.goal = goal
        self.map = occ_map
        self.dim = dim
        self.rbit = rbit
        self.m = no_samples
        self.ci = np.inf
        self.old_ci = np.inf
        self.cmin = np.linalg.norm(self.goal.np_arr - self.start.np_arr)
        self.flat_map = self.map.map.flatten()

        self.V = set()
        self.E = set()
        self.E_vis = set()
        self.x_new = set()
        self.x_reuse = set()
        self.unexpanded = set()
        self.vs = set()
        self.unconnected = set()
        self.vsol = set()

        self.qv = PriorityQueue()
        self.qe = PriorityQueue()
        self.qe_order = 0
        self.qv_order = 0

        self.V.add(start)
        self.unconnected.add(goal)
        self.unexpanded = self.V.copy()
        self.x_new = self.unconnected.copy()

        self.qv.put((start.gt + start.h_hat, self.qv_order, start))
        self.qv_order -= 1
        self.get_PHS()

        self.json_save_dir = (
            f"{os.path.abspath(os.path.dirname(__file__))}/../Logs/PyViz/"
            + str(datetime.now())
            + "/"
        )
        print(self.json_save_dir)
        os.makedirs(self.json_save_dir, exist_ok=True)
        self.json_contents = {"edges": [], "final_path": [], "ci": []}

    def gt(self, node):
        if node == self.start:
            return 0
        elif node not in self.V:
            return np.inf
        return node.par_cost + node.parent.gt

    def c_hat(self, node1, node2):
        return np.linalg.norm(node1.np_arr - node2.np_arr)

    def a_hat(self, node1, node2):
        return node1.g_hat + self.c_hat(node1, node2) + node2.h_hat

    def c(self, node1, node2, scale=10):
        x1, y1 = node1.tup
        x2, y2 = node2.tup

        n_divs = int(scale * np.linalg.norm(node1.np_arr - node2.np_arr))

        for lam in np.linspace(0, 1, n_divs):
            x = int(x1 + lam * (x2 - x1))
            y = int(y1 + lam * (y2 - y1))
            if (x, y) in self.map.occupied:
                return np.inf

        return self.c_hat(node1, node2)

    def near(self, search_set, node):
        near = set()
        for n in search_set:
            if (self.c_hat(n, node) <= self.rbit) and (n != node):
                near.add(n)
        return near

    def expand_next_vertex(self):
        #! Potential BUG: Somewhere here or in near when we add node we must send it with gt, parent, par_cost initialized.
        vmin = self.qv.get(False)[2]
        x_near = None
        if vmin in self.unexpanded:
            x_near = self.near(self.unconnected, vmin)
        else:
            intersect = self.unconnected & self.x_new
            x_near = self.near(intersect, vmin)

        for x in x_near:
            if self.a_hat(vmin, x) < self.ci:
                cost = vmin.gt + self.c(vmin, x) + x.h_hat
                self.qe.put((cost, self.qe_order, (vmin, x)))
                self.qe_order -= 1
                #! Potential BUG: Should look like this: self.qe.put((vmin.gt + self.c(vmin, x), x.h_hat), (Node(vmin), Node(x))))

        if vmin in self.unexpanded:
            v_near = self.near(self.V, vmin)
            for v in v_near:
                if (
                    (not (vmin, v) in self.E)
                    and (self.a_hat(vmin, v) < self.ci)
                    and (v.g_hat + self.c_hat(vmin, v) < v.gt)
                ):
                    cost = vmin.gt + self.c(vmin, v) + v.h_hat
                    self.qe.put((cost, self.qe_order, (vmin, v)))
                    self.qe_order -= 1
            self.unexpanded.remove(vmin)

    def sample_unit_ball(self):
        u = np.random.uniform(-1, 1, self.dim)
        norm = np.linalg.norm(u)
        r = np.random.random() ** (1.0 / self.dim)
        return r * u / norm

    def samplePHS(self):
        center = (self.start.np_arr + self.goal.np_arr) / 2
        a1 = (self.goal.np_arr - self.start.np_arr) / self.cmin
        one_1 = np.eye(a1.shape[0])[:, 0]
        U, S, Vt = np.linalg.svd(np.outer(a1, one_1.T))
        Sigma = np.diag(S)
        lam = np.eye(Sigma.shape[0])
        lam[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        cwe = np.matmul(U, np.matmul(lam, Vt))
        r1 = self.ci / 2
        rn = [np.sqrt(self.ci**2 - self.cmin**2) / 2] * (self.dim - 1)
        r = np.array([r1] + rn)

        while True:
            try:
                x_ball = self.sample_unit_ball()
                op = np.matmul(np.matmul(cwe, r), x_ball) + center
                op = np.around(op, 7)
                if (int(op[0]), int(op[1])) in self.intersection:
                    break
            except:
                print(op, x_ball, r, self.cmin, self.ci, cwe)
                exit()

        return op

    def get_PHS(self):
        # self.xphs = set(np.argwhere(self.map.f_hat_map <= self.ci))
        self.xphs = set([tuple(x) for x in np.argwhere(self.map.f_hat_map < self.ci)])
        # TODO: Why is self.old_ci being updated here?
        # self.old_ci = self.ci
        self.intersection = self.xphs & self.map.free

    def sample(self):
        xrand = None
        if self.old_ci != self.ci:
            self.get_PHS()

        if len(self.xphs) < len(self.flat_map):
            xrand = self.samplePHS()
        else:
            xrand = self.map.new_sample()

        #! Potential BUG: Same issue propagates here afaik.
        return Node(xrand)

    def prune(self):
        self.x_reuse = set()
        new_unconnected = set()
        for n in self.unconnected:
            # if n.f_hat >= self.ci:
            #     self.unconnected.remove(n)
            if n.f_hat < self.ci:
                new_unconnected.add(n)
        self.unconnected = new_unconnected

        sorted_nodes = sorted(self.V, key=lambda x: x.gt, reverse=True)
        for v in sorted_nodes:
            if v != self.start and v != self.goal:
                if (v.f_hat > self.ci) or (v.gt + v.h_hat > self.ci):
                    self.V.discard(v)
                    self.vsol.discard(v)
                    self.unexpanded.discard(v)
                    self.E.discard((v.parent, v))
                    self.E_vis.discard((v.parent.tup, v.tup))
                    v.parent.children.remove(v)
                    if v.f_hat < self.ci:
                        self.x_reuse.add(v)
                    else:
                        del v
        self.unconnected.add(self.goal)

    def remove_children(self, n):
        connected = list(n.children)
        if connected != []:
            for i in range(len(connected)):
                c = connected[i]
                # print("Removing", c)
                self.remove_children(c)
                c.parent = None
                c.par_cost = np.inf
                self.E.discard((n, c))
                self.E_vis.discard((n.tup, c.tup))
        if n in self.V and n != self.start and n != self.goal:
            # print(f"Discarding {n} from V")
            self.V.remove(n)
            self.pruned.add(n)

    def final_solution(self):
        if self.goal.gt == np.inf:
            return None, None
        path = []
        path_length = 0
        node = self.goal
        while node != self.start:
            path.append(node.tup)
            path_length += node.par_cost
            node = node.parent
        path.append(self.start.tup)
        return path[::-1], path_length

    def update_children_gt(self, node):
        for c in node.children:
            c.gt = c.par_cost + node.gt
            self.update_children_gt(c)

    def save_data(self):
        if self.json_contents["ci"] == []:
            self.json_contents["ci"].append(self.ci)
            current_solution, _ = self.final_solution()
            self.json_contents["final_path"].append(current_solution)

        if self.json_contents["ci"][-1] != self.ci:
            self.json_contents["ci"].append(self.ci)
            current_solution, _ = self.final_solution()
            self.json_contents["final_path"].append(current_solution)

        self.json_contents["edges"].append(list(self.E_vis))

    def dump_data(self, goal_num):
        print("Dumping data...")
        json_object = json.dumps(self.json_contents, indent=4)

        # Writing to sample.json
        with open(
            f"{self.json_save_dir}/path{goal_num:02d}.json",
            "w",
        ) as outfile:
            outfile.write(json_object)
        self.json_contents = {
            "edges": [],
            "final_path": [],
            "ci": [],
        }
        print("Data dumped")

    def make_plan(self, save=False):
        start = time.time()
        unchanged = 0
        it = 0
        goal_num = 0

        if self.start.tup not in self.map.free or self.goal.tup not in self.map.free:
            print("Start or Goal not in free space")
            return None, None

        if self.start.tup == self.goal.tup:
            print("Start and Goal are the same")
            self.vsol.add(self.start)
            self.ci = 0
            return [self.start.tup], 0

        try:
            while True:
                it += 1
                if self.qe.empty() and self.qv.empty():
                    self.prune()
                    if save:
                        self.save_data()

                    x_sample = set()

                    while len(x_sample) < self.m:
                        x_sample.add(self.sample())

                    self.x_new = self.x_reuse | x_sample
                    self.unconnected = self.unconnected | self.x_new
                    for n in self.V:
                        self.qv.put((n.gt + n.h_hat, self.qv_order, n))
                        self.qv_order -= 1
                while True:
                    if self.qv.empty():
                        break
                    self.expand_next_vertex()

                    if self.qe.empty():
                        continue
                    if self.qv.empty() or self.qv.queue[0][0] <= self.qe.queue[0][0]:
                        break

                if not (self.qe.empty()):
                    (vmin, xmin) = self.qe.get(False)[2]

                    if vmin.gt + self.c_hat(vmin, xmin) + xmin.h_hat < self.ci:
                        if vmin.gt + self.c_hat(vmin, xmin) < xmin.gt:
                            cedge = self.c(vmin, xmin)
                            if vmin.gt + cedge + xmin.h_hat < self.ci:
                                if vmin.gt + cedge < xmin.gt:
                                    if xmin in self.V:
                                        # tree.remove_edge(tree.parent(xmin), xmin)
                                        # tree.add_edge(vmin, xmin, weight=cedge)
                                        self.E.remove((xmin.parent, xmin))
                                        self.E_vis.remove((xmin.parent.tup, xmin.tup))
                                        xmin.parent.children.remove(xmin)

                                        xmin.parent = vmin
                                        xmin.par_cost = cedge
                                        xmin.gt = self.gt(xmin)
                                        self.E.add((xmin.parent, xmin))
                                        self.E_vis.add((xmin.parent.tup, xmin.tup))
                                        xmin.parent.children.add(xmin)
                                        self.update_children_gt(xmin)

                                    else:
                                        self.V.add(xmin)
                                        # self.add_edge(vmin, xmin, weight=cedge)
                                        xmin.parent = vmin
                                        xmin.par_cost = cedge
                                        xmin.gt = self.gt(xmin)
                                        self.E.add((xmin.parent, xmin))
                                        self.E_vis.add((xmin.parent.tup, xmin.tup))
                                        self.qv_order -= 1
                                        self.unexpanded.add(xmin)
                                        if xmin == self.goal:
                                            self.vsol.add(xmin)
                                        xmin.parent.children.add(xmin)
                                        self.unconnected.remove(xmin)
                                    # if self.goal.gt != self.ci:
                                    # self.old_ci = self.ci
                                    self.ci = max(self.goal.gt, self.cmin)

                                    if save:
                                        self.save_data()

                                    if self.ci != self.old_ci:
                                        print("\n\nGOAL FOUND ", goal_num)
                                        print("Time Taken:", time.time() - start)
                                        start = time.time()
                                        solution, length = self.final_solution()
                                        print("Path:", solution)
                                        print("Path Length:", length)
                                        print(
                                            f"Old CI: {self.old_ci}, New CI: {self.ci}, ci - cmin: {round(self.ci - self.cmin, 5)}, Difference in CI: {round(self.old_ci - self.ci, 5)}"
                                        )
                                        self.old_ci = self.ci
                                        if save:
                                            print("Dump")
                                            self.dump_data(goal_num)
                                        goal_num += 1

                    else:
                        self.qe = PriorityQueue()
                        self.qv = PriorityQueue()
                        unchanged += 1

                else:
                    self.qe = PriorityQueue()
                    self.qv = PriorityQueue()
                    unchanged += 1
            return self.final_solution()
        except KeyboardInterrupt:
            print(time.time() - start)
            print(self.final_solution())
            return self.final_solution()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    start_arr = np.array([0, 0])
    goal_arr = np.array([0, 99])

    start = Node((0, 0), gt=0)
    goal = Node((0, 99))

    map_path = (
        f"{os.path.abspath(os.path.dirname(__file__))}/../gridmaps/occupancy_map.png"
    )

    map_obj = Map(start=start, goal=goal, size=(100, 100))
    planner = bitstar(start=start, goal=goal, occ_map=map_obj, no_samples=100, rbit=10)
    path, path_length = planner.make_plan(save=True)
    print(planner.ci, planner.old_ci)
    print(path, path_length)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()
