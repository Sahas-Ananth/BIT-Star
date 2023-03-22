import networkx as nx
# from collections import PriorityQueue
from queue import PriorityQueue
import numpy as np
from PIL import Image
from bresenham import bresenham
import random
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)

class Map:
    def __init__(self, start, goal, image_path):
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.dim = 2

        # self.map = np.asarray(Image.open(image_path))
        self.map = image_path
        # return self.map

    def free_nodes(self):
        ind = np.where(self.map == 1)
        # print(list(zip(ind[0], ind[1])))
        return list(zip(ind[0], ind[1]))

    def occupied(self):
        ind = np.where(self.map == 0)
        return list(zip(ind[0], ind[1]))

    def sample(self):
        # Blame Shankara if this doesn't work.
        # copy_map = self.map.map.copy()
        ind = np.argwhere(self.map >= 0)
        return random.choices(ind)[0]


class Tree(nx.DiGraph):
    def __init__(self, start, goal, image_path):
        super(Tree, self).__init__()
        self.start = start
        self.add_node(start)
        self.goal = goal
        self.add_node(goal)
        self.qv = PriorityQueue()
        self.qe = PriorityQueue()
        self.rbit = 1.5  # Radius of ball of interest
        self.unexpanded = [self.start]
        self.x_new = self.unconnected()
        # print("X_NEW", self.x_new)
        # exit()
        self.x_reuse = []
        self.m = 3

        self.map = Map(start, goal, image_path)
        self.map_array = self.map.map


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
        # print("h_hat")
        return np.linalg.norm(np.array(node) - np.array(self.goal))

    def c_hat(self, node1, node2):  # Heuristic cost from node1 to node2.
        # print("c_hat")
        return np.linalg.norm(np.array(node1) - np.array(node2))
    

    def a_hat(self, node1, node2):  # function for lazy people
        return self.g_hat(node1) + self.c_hat(node1, node2) + self.h_hat(node2)
    
    def c(self, node1, node2):
        # use bresenham to check if there is an obstacle between node1 and node2
        # if there is an obstacle, return np.inf
        # else return c_hat
        cells = list(bresenham(node1[0], node1[1], node2[0], node2[1]))

        for cell in cells:
            if cell in self.map.occupied():
                # this could be slow if there are a lot of obstacles since occupied() could be O(n)
                return np.inf
            
        return self.c_hat(node1, node2)

    def parent(self, node):
        # print("Parent", node)
        # print("parent", list(self.predecessors(node)))
        return list(self.predecessors(node))[0]
    
    def connected(self):
        connected = [self.start]
        for n in self.nodes():
            # print('\n\n', n, self.out_degree(n), self.in_degree(n), '\n\n')
            if self.in_degree(n) > 0 and n != self.start:
                connected.append(n)

        return connected

    def unconnected(self):
        unconnected = []
        for n in self.nodes():
            # print('\n\n', n, self.out_degree(n), self.in_degree(n), '\n\n')
            if self.out_degree(n) == 0 and self.in_degree(n) == 0 and n != self.start:
                unconnected.append(n)

        return unconnected
        # return [
        #     n
        #     for n in self.nodes()
        #     if self.out_degree(n) == 0 and self.in_degree(n) == 0 and n != self.start
        # ]

    def near(self, search_list, node):
        # ? Possible bug here. "The function Near returns the states that meet the selected RGG connection criterion for a given vertex." We dont know what is "the RGG connection criterion".
        # print(self.unconnected())
        near = []
        # print("SEARCH_LIST", search_list)
        for search_node in search_list:
            if (self.c_hat(search_node, node) <= self.rbit) and (search_node != node):
                near.append(search_node)
        # print("Near uncon", near_uncon)
        return near

    def expand_next_vertex(self):
        # print("QV", self.qv.queue)
        vmin = self.qv.get(False)[1]
        # print("VMIN", vmin)
        # print("Nodes", self.nodes())
        if vmin in self.unexpanded:
            x_near = self.near(self.unconnected(), vmin)
        else:
            # print("UNCONNECTED", self.unconnected())
            # print("X_NEW", self.x_new)
            intersect = list(set(self.unconnected()) & set(self.x_new))
            # print("INTERSECTION", intersect)
            x_near = self.near(intersect, vmin)
        # print("X near", vmin, x_near)
        for x in x_near:    
            # print("X", x)
            if self.a_hat(vmin, x) < self.ci:
                self.qe.put((self.gt(vmin) + self.c_hat(vmin, x) + self.h_hat(x), (vmin, x)))
        if vmin in self.unexpanded:
            v_near = self.near(self.connected(), vmin)
            for v in v_near:
                # print("V", v)
                if (not self.has_edge(vmin, v))\
                    and (self.a_hat(vmin, v) < self.ci)\
                    and (self.g_hat(v) + self.c_hat(vmin, v) < self.gt(v)):
                    self.qe.put((self.gt(vmin) + self.c_hat(vmin, v) + self.h_hat(v), (vmin, v)))
            self.unexpanded.remove(vmin)

    def sample_unit_ball(self, d):
        # Method 20 in https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

        # Vector of length d
        u = np.random.uniform(-1, 1, d)
        # Normalize the vector
        norm = np.linalg.norm(u)

        # Pick a random radius within [0, 1) to the d^-1 power. (idk how this works)
        r = np.random.random() ** (1.0 / d)

        x = r * u / norm
        return x

    def samplePHS(self):
        # TODO The ending PHS value always returns (x, y) where x < y, which causes the path to stray downwards in case where the start and goal are diagonal from each other
        start, goal = np.array(self.start), np.array(self.goal)
        cmin = np.linalg.norm(goal - start)
        center = (start + goal) / 2
        a1 = (goal - start) / cmin
        one_1 = np.eye(a1.shape[0])[:, 0]
        # print("A!, one_1", a1, one_1, a1 @ one_1.T, np.outer(a1, one_1.T))
        U, S, Vt = np.linalg.svd(np.outer(a1, one_1.T))
        # print(U, S, Vt)
        Sigma = np.diag(S)
        lam = np.eye(Sigma.shape[0])
        lam[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        # print("LAM", lam)
        cwe = np.matmul(U, np.matmul(lam, Vt))
        # print("CWE", cwe)
        r1 = self.ci / 2
        # print("CI", self.ci, cmin, r1)
        rn = [np.sqrt(self.ci**2 - cmin**2) / 2] * (self.dim - 1)
        r = np.diag([r1] + rn)
        # print("R", r)
        xball = self.sample_unit_ball(self.dim)
        phs = np.matmul(np.matmul(cwe, r), xball) + center

        # TODO the issue mentioned above can be demonstrated here. With this if condition active, it never prints.
        # if phs[0]< 0 or phs[1] < 0:
        # print("XBALL", xball, xball[0]**2 + xball[1]**2)
        # print("PHS", phs, self.ci)
        return np.matmul(np.matmul(cwe, r), xball) + center
        # return np.matmul(cwe, np.matmul(r, xball)) + center

    def sample(self):
        xrand = None
        mapx, mapy = self.map_array.shape
        xphs = []
        for x in range(mapx):
            for y in range(mapy):
                if self.f_hat(np.array([x, y])) < self.ci:
                    xphs.append((x, y))
                    # break
        # xphs = [
        #     (x, y)
        #     for x in range(mapx)
        #     for y in range(mapy)
        #     if self.f_hat(np.array([x, y])) < self.ci
        # ]
        # exit()]
        while True:
            copy_map = self.map.map.copy()

            if len(xphs) < len(copy_map.flatten()):
                # print("PHS")
                xrand = self.samplePHS()
            else:
                xrand = self.map.sample()
            # if int(xrand[0]) == int(xrand[1]):
            #     print("XRAND", (int(xrand[0]), int(xrand[1])))
            # print("XPHS & FREE", len(list(set(xphs) & set(self.map.free_nodes()))))

            # demo = np.ones((10, 10))
            # for x in list(set(xphs) & set(self.map.free_nodes())):
            #     demo[x] = 0
            # print(demo)
            # exit()
                  
            # exit()
            xrand = (int(xrand[0]), int(xrand[1])) 

            if xrand in list(set(xphs) & set(self.map.free_nodes())):
                break

        return xrand

    def prune(self):
        self.x_reuse = []
        for n in self.unconnected():
            if self.f_hat(n) >= self.ci:
                self.remove_node(n)
        # print("nodes", self.nodes())
        sorted_nodes = sorted(self.connected(), key=self.gt)
        # print("edges", self.edges)
        # print("Sorted nodes", sorted_nodes)
        # # print("gt")
        # for node in sorted_nodes:
        #     print(node, self.gt(node))
        # exit()
        for v in sorted_nodes:
            if v != self.goal and v != self.start:
                if (self.f_hat(v) > self.ci) or (self.gt(v) + self.h_hat(v) > self.ci):
                    self.remove_node(v)
                    # self.vsol.remove(v)
                    # print("Unexpanded", self.unexpanded, v)
                    self.unexpanded.remove(v)
                    if self.f_hat(v) < self.ci:
                        self.x_reuse.append(v)
        return self.x_reuse
    
    def final_solution(self):
        path = []
        node = self.goal
        # nx.draw(self)
        # plt.show()
        while node!=self.start:
            path.append(node)
            node = self.parent(node)
        return path[::-1]

def bitstar():

    start = (0,0)
    goal = (9,9)
    unchanged = 0

    map = np.ones((10,10))

    tree = Tree(start=start, goal=goal, image_path=map)

    iteration = 0
    while True:

        # print the iteration number
        print("Iteration: ", iteration)
        iteration += 1
        # print("CI", tree.ci)
        
        if tree.qe.empty() and tree.qv.empty():
            # print("Sampling")
            tree.x_reuse = tree.prune()

            # TODO: sample m nodes
            x_sampling = []
            while len(x_sampling) <= tree.m:
                sample = tree.sample()
                # print("Sampling", x_sampling)
                if sample not in x_sampling:
                    x_sampling.append(sample)
                    # print("Appended", x_sampling)
            

            # print("x_reuse", tree.x_reuse, x_sampling)
            tree.x_new = [*set(tree.x_reuse + x_sampling)]

            # print("x_new", tree.x_new)
            # tree.
            # print("x_sampling", x_sampling)
            tree.add_nodes_from(x_sampling)
            # print("Nodes", tree.nodes())
            for n in tree.connected():
                tree.qv.put((tree.gt(n) + tree.h_hat(n), n)) # is goal a part of V
            # print("queuev" , tree.qv.queue)
            # print("queuee" , tree.qe.queue)
        # return the value in the queue with the lowest cost but dont remove it

        # print(tree.qv.queue[0])
        # print(tree.qe.queue[0])

        while True:
            if tree.qv.empty():
                break
            tree.expand_next_vertex()
            # print("queuev" , tree.qv.queue)
            # print("queuee" , tree.qe.queue)
            if tree.qv.empty() or tree.qe.empty() or tree.qv.queue[0][0] <= tree.qe.queue[0][0]:
                break

        if(not(tree.qe.empty())):

            (vmin, xmin) = tree.qe.get(False)[1]
            # print("Vmin, Xmin", vmin, xmin)

            if tree.gt(vmin) + tree.c_hat(vmin, xmin) + tree.h_hat(xmin) < tree.ci:
                # print("2nd if")
                if tree.gt(vmin) + tree.c_hat(vmin, xmin) < tree.gt(xmin):
                    
                    # do we check if the edge is already in the occupied space?
                    # c_hat = c?
                    cedge = tree.c(vmin, xmin)
                    # print("3rd if")
                    if tree.gt(vmin) + cedge + tree.h_hat(xmin) < tree.ci:

                        # print("4th if")
                        
                        if tree.gt(vmin) + cedge < tree.gt(xmin):

                            if xmin in tree.connected():
                                tree.remove_edge(tree.parent(xmin), xmin)

                            else:
                                tree.qv.put((tree.gt(xmin) + tree.h_hat(xmin), xmin))
                                tree.unexpanded.append(xmin)
                                if xmin == tree.goal:
                                    tree.vsol.append(xmin)
                                    

                            tree.add_edge(vmin, xmin, weight=cedge)
                            # print("edge added")
                            tree.ci = tree.gt(tree.goal)
                            if xmin == tree.goal:
                                print("GOAL FOUND")
                                print(tree.final_solution())
                                # print(tree.edges)
                                # exit()
            else:
                tree.qe = PriorityQueue()
                tree.qv = PriorityQueue()
                unchanged += 1

            
        
        else:
            tree.qe = PriorityQueue()
            tree.qv = PriorityQueue()
            unchanged += 1

        if unchanged == 100:
            print("BREAK")
            break
        

        
    return tree.final_solution()



if __name__ == "__main__":
    path = bitstar()
    print("FINAL PATH", path)

# TODO: Write the Actual BIT* Algorithm.

