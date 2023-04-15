import argparse
from bit_star_opt_opt import *
import bit_star_opt_opt
from vis2 import *




def main(image_path, vis, start, goal, rbit, samples, dim, seed, output_path):

    random.seed(seed)
    np.random.seed(seed)

    start = []
    goal = []
    for i in range(opt.dim):
        start.append(int(opt.start[i]))
        goal.append(int(opt.goal[i]))
    start = np.array(start)
    goal = np.array(goal)

    if image_path is None:
        occ_map = np.ones((100, 100))
    else:
        map_path = (
            f"{os.path.abspath(os.path.dirname(__file__))}/{image_path}"
        )
        occ_map = np.array(Image.open(map_path))

    bit_star_opt_opt.start_arr = start
    bit_star_opt_opt.goal_arr = goal

    start_node = Node(tuple(start), gt=0)
    goal_node = Node(tuple(goal))



    map_obj = Map(start=start_node, goal=goal_node, occ_grid=occ_map)
    planner = bitstar(start=start_node, goal=goal_node, occ_map=map_obj, no_samples=samples, rbit=rbit, dim=dim)
    path, path_length = planner.make_plan(save=vis)

    print(planner.ci, planner.old_ci)
    print(path, path_length)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default=None, help='Path to the image file. If none, will default to a 100x100 empty grid')
    parser.add_argument('--vis', action='store_true', help='Whether or not to save and visualize outputs')
    parser.add_argument('--start', nargs='+', help='Start coordinates')
    parser.add_argument('--goal', nargs='+', help='Goal coordinates')
    parser.add_argument('--rbit', type=float, default=10, help='Edge maximum length')
    parser.add_argument('--samples', type=int, default=50, help='Number of new samples per iteration')
    parser.add_argument('--dim', type=int, default=2, help='Dimensions of working space')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--output_path', type=str, default=f"{os.path.abspath(os.path.dirname(__file__))}/Logs/PyViz/")

    opt = parser.parse_args()
    return opt



if __name__ == '__main__':
    opt = parse_opt()
    print(opt)

    assert(len(opt.start) == opt.dim)
    assert(len(opt.goal) == opt.dim)

    


    main(**vars(opt))
