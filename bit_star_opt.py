import networkx as nx

# from collections import PriorityQueue
from queue import PriorityQueue
import numpy as np
from PIL import Image
from bresenham import bresenham
import random
import matplotlib.pyplot as plt
import cv2
import cProfile

random.seed(0)
np.random.seed(0)


class Map:
    def __init__(self, start, goal, image_path):
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.dim = 2
        self.map = np.asarray(Image.open(image_path))

        ind = np.where(self.map > 0)
        self.free = list(zip(ind[0], ind[1]))

        ind = np.where(self.map == 0)
        self.occ = list(zip(ind[0], ind[1]))

    def free_nodes(self):
        return self.free

    def occupied(self):
        return self.occ

    def sample(self):
        while True:
            x = np.random.uniform(low=0, high=self.map.shape[0])
            y = np.random.uniform(low=0, high=self.map.shape[1])
            if self.map[int(x), int(y)]:
                return (x, y)


class Tree(nx.DiGraph):
    def __init__(self, start, goal, image_path):
        super(Tree, self).__init__()
        self.start = start
        self.add_node(start)
        self.goal = goal
        self.add_node(goal)
        self.qv = PriorityQueue()
        self.qe = PriorityQueue()
        self.rbit = 100  # Radius of ball of interest
        self.unexpanded = [self.start]
        self.x_new = self.unconnected()

        self.x_reuse = []
        self.m = 20
        self.V = [self.start]

        self.map = Map(start, goal, image_path)
        self.map_array = self.map.map
        self.copy_map_flat = self.map.map.copy().flatten()

        self.dim = 2  # Search space dimension

        self.qv.put((self.gt(start) + self.h_hat(start), start))

        self.vsol = []
        self.ci = np.inf
        self.old_ci = self.ci

        mapx, mapy = self.map_array.shape
        self.xphs = []
        for x in range(mapx):
            for y in range(mapy):
                if self.f_hat(np.array([x, y])) < self.ci:
                    self.xphs.append((x, y))
        self.intersection = list(set(self.xphs) & set(self.map.free_nodes()))

        if self.start == self.goal:
            self.vsol = [self.start]
            self.ci = 0

    def gt(self, node):  # Cost to traverse the tree from the root to node v
        # TODO fix parents of nodes. Currently throwing an error after first solution is found
        # Most likely due to when we prune the tree, nodes are still erroneosly kept as connected

        if node == self.start:
            return 0
        if node not in self.connected():
            return np.inf
        length = 0
        try:
            parent = self.parent(node)
        except:
            print(node in self.connected())
            print(node, self[node])
            print(self.predecessors(node))
        while parent != self.start:
            length += self[parent][node]["weight"]
            node = parent
            parent = self.parent(node)

        return length

    def f_hat(self, node):  # Total Heuristic cost of node
        return self.g_hat(node) + self.h_hat(node)

    def g_hat(self, node):  # Heuristic cost from start to node.
        return np.linalg.norm(np.array(node) - np.array(self.start))

    def h_hat(self, node):  # Heuristic cost from node to goal.
        return np.linalg.norm(np.array(node) - np.array(self.goal))

    def c_hat(self, node1, node2):  # Heuristic cost from node1 to node2.
        return np.linalg.norm(np.array(node1) - np.array(node2))

    def a_hat(self, node1, node2):  # function for lazy people
        return self.g_hat(node1) + self.c_hat(node1, node2) + self.h_hat(node2)

    def c(self, node1, node2):
        x1, y1 = node1
        x2, y2 = node2

        n_divs = int(10 * np.linalg.norm(np.array(node1) - np.array(node2)))

        for lam in np.linspace(0, 1, n_divs):
            x = int(x1 + lam * (x2 - x1))
            y = int(y1 + lam * (y2 - y1))
            if self.map.map[x, y] == 0:
                return np.inf
        return self.c_hat(node1, node2)

    def parent(self, node):
        return list(self.predecessors(node))[0]

    def connected(self):
        # connected = [self.start]
        # for n in self.nodes():
        #     if self.in_degree(n) > 0 and n != self.start:
        #         connected.append(n)

        # return connected
        return self.V

    def unconnected(self):
        unconnected = []
        for n in self.nodes():
            if self.out_degree(n) == 0 and self.in_degree(n) == 0 and n != self.start:
                unconnected.append(n)

        return unconnected

    def near(self, search_list, node):
        # ? Possible bug here. "The function Near returns the states that meet the selected RGG connection criterion for a given vertex." We dont know what is "the RGG connection criterion".

        near = []
        for search_node in search_list:
            if (self.c_hat(search_node, node) <= self.rbit) and (search_node != node):
                near.append(search_node)
        return near

    def expand_next_vertex(self):
        vmin = self.qv.get(False)[1]

        if vmin in self.unexpanded:
            x_near = self.near(self.unconnected(), vmin)
        else:
            intersect = list(set(self.unconnected()) & set(self.x_new))
            # print("INTERSECTION", intersect)
            x_near = self.near(intersect, vmin)
        # print("X near", vmin, x_near)
        for x in x_near:
            # print("X", x)
            if self.a_hat(vmin, x) < self.ci:
                self.qe.put(
                    (self.gt(vmin) + self.c_hat(vmin, x) + self.h_hat(x), (vmin, x))
                )
        if vmin in self.unexpanded:
            v_near = self.near(self.connected(), vmin)
            for v in v_near:
                # print("V", v)
                if (
                    (not self.has_edge(vmin, v))
                    and (self.a_hat(vmin, v) < self.ci)
                    and (self.g_hat(v) + self.c_hat(vmin, v) < self.gt(v))
                ):
                    self.qe.put(
                        (self.gt(vmin) + self.c_hat(vmin, v) + self.h_hat(v), (vmin, v))
                    )
            self.unexpanded.remove(vmin)

    def sample_unit_ball(self, d):
        # Method 20 in https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

        u = np.random.uniform(-1, 1, d)
        norm = np.linalg.norm(u)

        r = np.random.random() ** (1.0 / d)

        x = r * u / norm
        return x

    def samplePHS(self):
        start, goal = np.array(self.start), np.array(self.goal)
        cmin = np.linalg.norm(goal - start)
        center = (start + goal) / 2
        a1 = (goal - start) / cmin
        one_1 = np.eye(a1.shape[0])[:, 0]
        U, S, Vt = np.linalg.svd(np.outer(a1, one_1.T))
        Sigma = np.diag(S)
        lam = np.eye(Sigma.shape[0])
        lam[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        cwe = np.matmul(U, np.matmul(lam, Vt))
        r1 = self.ci / 2
        rn = [np.sqrt(self.ci**2 - cmin**2) / 2] * (self.dim - 1)
        r = np.diag([r1] + rn)
        xball = self.sample_unit_ball(self.dim)
        phs = np.matmul(np.matmul(cwe, r), xball) + center

        output = np.matmul(np.matmul(cwe, r), xball) + center
        output = list(np.around(np.array(output), 7))
        return output

    def sample(self):
        xrand = None
        if self.old_ci != self.ci:
            mapx, mapy = self.map_array.shape
            self.xphs = []
            for x in range(mapx):
                for y in range(mapy):
                    if self.f_hat(np.array([x, y])) < self.ci:
                        self.xphs.append((x, y))
            self.old_ci = self.ci
            self.intersection = list(set(self.xphs) & set(self.map.free_nodes()))

        while True:
            if len(self.xphs) < len(self.copy_map_flat):
                # print("PHS")
                xrand = self.samplePHS()
            else:
                xrand = self.map.sample()
            if (int(xrand[0]), int(xrand[1])) in self.intersection:
                break

        return tuple(xrand)

    def prune(self):
        # TODO potentially iterate through the connected nodes of nodes that are removed
        # and subsequently remove them from self.V to avoid potential errors.
        self.x_reuse = []
        for n in self.unconnected():
            if self.f_hat(n) >= self.ci:
                self.remove_node(n)
        # print("nodes", self.nodes())
        sorted_nodes = sorted(self.connected(), key=self.gt)

        for v in sorted_nodes:
            if v != self.goal and v != self.start:
                if (self.f_hat(v) > self.ci) or (self.gt(v) + self.h_hat(v) > self.ci):
                    self.remove_node(v)
                    self.V.remove(v)
                    # self.vsol.remove(v)
                    # print("Unexpanded", self.unexpanded, v)
                    if v in self.unexpanded:
                        self.unexpanded.remove(v)
                    if self.f_hat(v) < self.ci:
                        self.x_reuse.append(v)
        return self.x_reuse

    def final_solution(self):
        path = []
        node = self.goal
        while node != self.start:
            path.append(node)
            node = self.parent(node)
        path.append(self.start)
        return path[::-1]


def bitstar():
    start = (635, 140)
    goal = (350, 400)
    unchanged = 0

    maps = "gridmaps/occupancy_map.png"
    mp = (cv2.imread(maps, 0) / 255).astype(np.uint8)
    print(mp[256, 111])
    # plt.imshow(mp, cmap="gray")
    # plt.show()
    tree = Tree(start=start, goal=goal, image_path=maps)

    iteration = 0
    try:
        while True:
            # print("Iteration: ", iteration)
            iteration += 1

            if tree.qe.empty() and tree.qv.empty():
                tree.x_reuse = tree.prune()

                x_sampling = []
                while len(x_sampling) <= tree.m:
                    sample = tree.sample()
                    if sample not in x_sampling:
                        x_sampling.append(sample)
                        # print("Appeded", x_sampling)

                tree.x_new = [*set(tree.x_reuse + x_sampling)]

                tree.add_nodes_from(x_sampling)
                for n in tree.connected():
                    tree.qv.put((tree.gt(n) + tree.h_hat(n), n))  # is goal a part of V

            while True:
                if tree.qv.empty():
                    break
                tree.expand_next_vertex()

                if tree.qe.empty():
                    continue
                if tree.qv.empty() or tree.qv.queue[0][0] <= tree.qe.queue[0][0]:
                    break

            if not (tree.qe.empty()):
                (vmin, xmin) = tree.qe.get(False)[1]

                if tree.gt(vmin) + tree.c_hat(vmin, xmin) + tree.h_hat(xmin) < tree.ci:
                    if tree.gt(vmin) + tree.c_hat(vmin, xmin) < tree.gt(xmin):
                        cedge = tree.c(vmin, xmin)
                        if tree.gt(vmin) + cedge + tree.h_hat(xmin) < tree.ci:
                            if tree.gt(vmin) + cedge < tree.gt(xmin):
                                if xmin in tree.connected():
                                    tree.remove_edge(tree.parent(xmin), xmin)
                                    tree.add_edge(vmin, xmin, weight=cedge)
                                else:
                                    tree.V.append(xmin)
                                    tree.add_edge(vmin, xmin, weight=cedge)
                                    tree.qv.put(
                                        (tree.gt(xmin) + tree.h_hat(xmin), xmin)
                                    )
                                    tree.unexpanded.append(xmin)
                                    if xmin == tree.goal:
                                        tree.vsol.append(xmin)

                                tree.ci = tree.gt(tree.goal)
                                if xmin == tree.goal:
                                    print("GOAL FOUND")
                                    print(tree.final_solution())
                                    solution = tree.final_solution()
                                    # plot all the points in the solution
                                    extent = [
                                        0,
                                        tree.map_array.shape[1],
                                        tree.map_array.shape[0],
                                        0,
                                    ]
                                    # return solution
                                    plt.imshow(
                                        tree.map_array,
                                        cmap="gray",
                                        interpolation="nearest",
                                        extent=extent,
                                    )
                                    x, y = zip(*solution)
                                    plt.plot(y, x, "-rx")
                                    plt.grid()
                                    plt.show()
                                    return solution

                else:
                    tree.qe = PriorityQueue()
                    tree.qv = PriorityQueue()
                    unchanged += 1

            else:
                tree.qe = PriorityQueue()
                tree.qv = PriorityQueue()
                unchanged += 1

        # if unchanged == 100:

    except KeyboardInterrupt:
        print("BREAK")
        print(tree.ci)
        solution = tree.final_solution()

        extent = [0, tree.map_array.shape[1], tree.map_array.shape[0], 0]
        plt.imshow(
            tree.map_array,
            cmap="gray",
            interpolation="nearest",
            extent=extent,
        )
        x, y = zip(*solution)
        plt.plot(y, x, "-rx")
        plt.grid()
        plt.show()

    return tree.final_solution()


if __name__ == "__main__":
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()
    path = bitstar()
    print("FINAL PATH", path)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()
