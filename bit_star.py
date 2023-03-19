import networkx as nx
from collections import PriorityQueue
import numpy as np
from PIL import Image


class Map:
    def __init__(self, start, goal, path):
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.dim = 2

        self.map = np.asarray(Image.open(path))
        return self.map

    def free_nodes(self):
        return self.map[self.map == 1].tolist()

    def occupied(self):
        return self.map[self.map == 0].tolist()

    def sample(self):
        # Blame Shankara if this doesn't work.
        return np.random.choice(self.free_nodes())


class Tree(nx.DiGraph):
    def __init__(self, start, goal, path):
        super(Tree, self).__init__()
        self.start = start
        self.add_node(start)
        self.goal = goal
        self.add_node(goal)
        self.qv = PriorityQueue()
        self.qe = PriorityQueue()
        self.rbit = 10  # Radius of ball of interest
        self.unexpanded = [self.start]
        self.x_new = self.unconnected()
        self.x_reuse = []

        self.map = Map(start, goal, path)

        self.dim = 2  # Search space dimension

        self.qv.put((self.gt(start) + self.h_hat(start), start))

        self.vsol = []
        self.ci = np.inf

        if self.start == self.goal:
            self.vsol = [self.start]
            self.ci = 0

    def gt(self, node):  # Cost to traverse the tree from the root to node v
        try:
            return nx.astar_path_length(
                G=self,
                source=self.start,
                target=node,
                heuristic=self.c_hat,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            return np.inf

    def f_hat(self, node):  # Total Heuristic cost of node
        return self.g_hat(node) + self.h_hat(node)

    def g_hat(self, node):  # Heuristic cost from start to node.
        # return l2 norm of node between node and root
        return np.linalg.norm(np.array(node) - np.array(self.start))

    def h_hat(self, node):  # Heuristic cost from node to goal.
        return np.linalg.norm(np.array(node) - np.array(self.goal))

    def c_hat(self, node1, node2):  # Heuristic cost from node1 to node2.
        return np.linalg.norm(np.array(node1) - np.array(node2))

    def a_hat(self, node1, node2):  # function for lazy people
        return self.g_hat(node1) + self.c_hat(node1, node2) + self.h_hat(node2)

    def parent(self, node):
        return list(self.predecessors(node))[0]

    def unconnected(self):
        return [
            n
            for n in self.nodes()
            if self.out_degree(n) == 0 and self.in_degree(n) == 0
        ]

    def near(self, node):
        # ? Possible bug here. "The function Near returns the states that meet the selected RGG connection criterion for a given vertex." We dont know what is "the RGG connection criterion".
        return [
            unconn
            for unconn in self.unconnected()
            if self.c_hat(unconn, node) <= self.rbit
        ]

    def expand_next_vertex(self):
        vmin = self.qv.get()
        if vmin in self.unexpanded:
            x_near = self.near(vmin)
        else:
            x_near = self.near([n for n in self.x_new if n in self.unconnected()])

        self.qe.put(
            (self.a_hat(vmin, x), (vmin, x))
            for x in x_near
            if self.a_hat(vmin, x) < self.ci
        )

        if vmin in self.unexpanded:
            v_near = self.near(self.nodes())
            self.qe.put(
                (self.a_hat(vmin, v), (vmin, v))
                for v in v_near
                if (not self.has_edge(vmin, v))
                and (self.a_hat(vmin, v) < self.ci)
                and (self.g_hat(v) + self.c_hat(vmin, v) < self.gt(v))
            )
            self.unexpanded.remove(vmin)

    def sample_unit_ball(self, d):
        # Method 20 in https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

        # Vector of length d
        u = np.random.uniform(0, 1, d)
        # Normalize the vector
        norm = np.linalg.norm(u)

        # Pick a random radius within [0, 1) to the d^-1 power. (idk how this works)
        r = np.random.random() ** (1.0 / d)

        x = r * u / norm
        return x

    def samplePHS(self):
        start, goal = np.array(self.start), np.array(self.goal)
        cmin = np.linalg.norm(goal - start)
        center = (start + goal) / 2
        a1 = goal - start / cmin
        one_1 = np.eye(a1.shape[0])[:, 0]
        U, S, V = np.linalg.svd(a1 @ one_1.T)

        lam = np.eye(S.shape[0])
        lam[-1, -1] = np.linalg.det(U) * np.linalg.det(V.T)
        cwe = np.matmul(U, np.matmul(lam, V))
        r1 = self.ci / 2
        rn = [np.sqrt(self.ci**2 - cmin**2) / 2] * (self.dim - 1)
        r = np.diag([r1] + rn)

        xball = self.sample_unit_ball(self.dim)
        return np.matmul(cwe, np.matmul(r, xball)) + center

    def sample(self):
        xrand = None
        mapx, mapy = self.map.shape
        xphs = [
            (x, y)
            for x in range(mapx)
            for y in range(mapy)
            if self.f_hat(np.array([x, y])) < self.ci
        ]

        while True:
            if len(xphs) < len(self.map.free_nodes()):
                xrand = self.samplePHS()
            else:
                xrand = self.map.sample()
            if xrand in [xphs + self.map.free_nodes()]:
                break

        return xrand

    def prune(self):
        self.x_reuse = []
        _ = [self.remove_node(n) for n in self.unconnected() if self.f_hat(n) > self.ci]
        sorted_nodes = sorted(self.nodes(), key=self.gt)
        for v in sorted_nodes:
            if (self.f_hat(v) > self.ci) or (self.gt(v) + self.h_hat(v) > self.ci):
                self.remove_node(v)
                self.vsol.remove(v)
                self.unexpanded.remove(v)
                if self.f_hat(v) < self.ci:
                    self.x_reuse.append(v)
        return self.x_reuse


# TODO: Write the Actual BIT* Algorithm.
