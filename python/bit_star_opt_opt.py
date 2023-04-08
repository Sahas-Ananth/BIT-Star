#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from queue import PriorityQueue
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import cv2
import cProfile
import time

random.seed(0)
np.random.seed(0)

global start
global goal


class Node:
    def __init__(self, x, y, parent=None, gt=np.inf, par_cost=None):
        self.x = x
        self.y = y
        self.tup = (x, y)
        self.np_arr = np.array([x, y])
        self.g_hat = self.gen_g_hat()
        self.h_hat = self.gen_h_hat()
        self.f_hat = self.g_hat + self.h_hat
        # TODO: Discuss with group about having cost from parent to node. This was weight parameter in the spaghetti code
        # assert par_cost is not None
        # assert gt is not None
        # assert parent is not None
        self.parent = parent
        self.par_cost = par_cost
        self.gt = gt

        global start
        self.start = start.np_arr
        global goal
        self.goal = goal.np_arr

    def gen_g_hat(self):
        return np.linalg.norm(self.np_arr - self.start)

    def gen_h_hat(self):
        return np.linalg.norm(self.np_arr - self.goal)

    def __str__(self) -> str:
        return str(self.tup)

    def __repr__(self) -> str:
        return str(self.tup)


class Map:
    def __init__(self, start, goal, image_path):
        self.start = start
        self.goal = goal
        self.obstacles = set()
        self.dim = 2
        self.map = np.array(Image.open(image_path))
        ind = np.argwhere(self.map > 0)
        self.free = set(zip(ind[0], ind[1]))
        ind = np.argwhere(self.map == 0)
        self.occupied = set(zip(ind[0], ind[1]))
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
            noise = np.random.uniform(-1, 1, self.dim)
            new_node = free_node + noise
            if (int(new_node[0]), int(new_node[1])) in self.free:
                return new_node

    def get_f_hat_map(self):
        map_x, map_y = self.map.shape
        self.f_hat_map = np.zeros((map_x, map_y))
        for x in range(map_x):
            for y in range(map_y):
                #! Potential BUG: Possible bug here with the Node class not having gt, parent, par_cost initialized.
                self.f_hat_map[x, y] = Node(x, y).f_hat


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
        self.flat_map = self.map.map.flatten()

        self.V = set()
        self.E = set()
        self.x_new = set()
        self.x_reuse = set()
        self.unexpanded = set()
        self.vs = set()
        self.unconnected = set()

        self.qv = PriorityQueue()
        self.qe = PriorityQueue()

        self.V.add(start)
        self.unconnected.add(goal)
        self.unexpanded = self.V.copy()
        self.x_new = self.unconnected.copy()

        self.qv.put((start.gt + start.h_hat, start))
        self.get_PHS()

    def gt(self, node):
        if node == self.start:
            return 0
        elif node not in self.V:
            return np.inf

        length = 0
        while node != self.start:
            length += node.par_cost
            node = node.parent

        return length

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
        vmin = self.qv.get(False)[1]
        x_near = None
        if vmin in self.unexpanded:
            x_near = self.near(self.unconnected, vmin)
        else:
            intersect = self.unconnected & self.x_new
            x_near = self.near(intersect, vmin)

        for x in x_near:
            if self.a_hat(vmin, x) < self.ci:
                self.qe.put((vmin.gt + self.c(vmin, x), x.h_hat), (vmin, x))
                #! Potential BUG: Should look like this: self.qe.put((vmin.gt + self.c(vmin, x), x.h_hat), (Node(vmin), Node(x))))

        if vmin in self.unexpanded:
            v_near = self.near(self.V, vmin)
            for v in v_near:
                if (
                    (not (vmin, v) in self.E)
                    and (self.a_hat(vmin, v) < self.ci)
                    and (v.g_hat + self.c_hat(vmin, v) < v.gt)
                ):
                    self.qe.put((vmin.gt + self.c_hat(vmin, v) + v.h_hat), (vmin, v))
            self.unexpanded.remove(vmin)

    def sample_unit_ball(self):
        u = np.random.uniform(-1, 1, self.dim)
        norm = np.linalg.norm(u)
        r = np.random.random() ** (1.0 / self.dim)
        return r * u / norm

    def samplePHS(self):
        cmin = np.linalg.norm(self.goal.np_arr - self.start.np_arr)
        center = (self.start.np_arr + self.goal.np_arr) / 2
        a1 = (self.goal.np_arr - self.start.np_arr) / cmin
        one_1 = np.eye(a1.shape[0])[:, 0]
        U, S, Vt = np.linalg.svd(np.outer(a1, one_1.T))
        Sigma = np.diag(S)
        lam = np.eye(Sigma.shape[0])
        lam[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        cwe = np.matmul(U, np.matmul(lam, Vt))
        r1 = self.ci / 2
        rn = [np.sqrt(self.ci**2 - cmin**2) / 2] * (self.dim - 1)
        r = np.array([r1] + rn)

        while True:
            x_ball = self.sample_unit_ball()
            op = np.matmul(np.matmul(cwe, r), x_ball) + center
            op = np.around(op, 7)
            if (int(op[0]), int(op[1])) in self.intersection:
                break

        return op

    def get_PHS(self):
        map_x, map_y = self.map.shape
        self.xphs = set(zip(*np.argwhere(self.map.f_hat_map <= self.ci)))
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
            xrand = self.map.sample()

        #! Potential BUG: Same issue propagates here afaik.
        return Node(xrand)

    def prune(self):
        self.x_reuse = set()
        for n in self.unconnected:
            if n.f_hat >= self.ci:
                self.unconnected.remove(n)

        sorted_nodes = sorted(self.V, key=lambda x: x.gt)
        for v in sorted_nodes:
            if v != self.start and v != self.goal:
                if (v.f_hat >= self.ci) or (v.gt + v.h_hat >= self.ci):
                    self.V.discard(v)
                    self.vsol.discard(v)
                    self.unexpanded.discard(v)
                    self.E.discard((v.parent, v))
                    if v.f_hat < self.ci:
                        self.x_reuse.add(v)

    def final_solution(self):
        path = []
        path_length = 0
        node = self.goal
        while node != self.start:
            path.append(node.tup)
            path_length += node.par_cost
            node = node.parent
        path.append(self.start.tup)
        return path[::-1], path_length

    def make_plan(self):
        start = time.time()

        if self.start.tup not in self.map.free or self.goal.tup not in self.map.free:
            return None, None

        if self.start.tup == self.goal.tup:
            self.vsol.add(self.start)
            self.ci = 0
            return [self.start.tup], 0

        while self.ci <= self.old_ci:
            if self.qe.empty() and self.qv.empty():
                self.prune()
                x_sample = set()
                while len(x_sample) < self.m:
                    x_sample.add(self.sample())

                #! Potential BUG fix: iterate over x_new and set it as Node(tuple, gt, parent, par_cost)
                self.x_new = self.x_reuse | x_sample
                self.unconnected = self.x_new
                for n in self.V:
                    self.qv.put((n.gt + n.h_hat, n))

            # TODO: Continue after this.
            raise NotImplementedError


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    start = Node(635, 140)
    goal = Node(350, 400)
    map_path = "../gridmaps/occupancy_map.png"

    map_obj = Map(start=start, goal=goal, map_path=map_path)
    planner = bitstar(start=start, goal=goal, occ_map=map_obj)
    path, path_length = planner.make_plan()
