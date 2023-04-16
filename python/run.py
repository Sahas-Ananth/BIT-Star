import argparse
from bit_star_opt_opt import *
import bit_star_opt_opt
from vis2 import *
import sys


def main(map_name, vis, start, goal, rbit, samples, dim, seed, stop_time):
    pwd = os.path.abspath(os.path.dirname(__file__))

    random.seed(seed)
    np.random.seed(seed)

    start = []
    goal = []
    for i in range(opt.dim):
        start.append(float(opt.start[i]))
        goal.append(float(opt.goal[i]))
    start = np.array(start)
    goal = np.array(goal)

    log_dir = f"{pwd}/../Logs/{map_name}"
    
    os.makedirs(log_dir, exist_ok=True)
   

    map_path = f"{pwd}/../gridmaps/{map_name}"
    occ_map = np.array(Image.open(map_path))

    bit_star_opt_opt.start_arr = start
    bit_star_opt_opt.goal_arr = goal

    start_node = Node(tuple(start), gt=0)
    goal_node = Node(tuple(goal))

    map_obj = Map(start=start_node, goal=goal_node, occ_grid=occ_map)
    planner = None
    if vis:
        planner = bitstar(
            start=start_node,
            goal=goal_node,
            occ_map=map_obj,
            no_samples=samples,
            rbit=rbit,
            dim=dim,
            log_dir=log_dir,
            stop_time=stop_time,
        )
    else:
        planner = bitstar(
            start=start_node,
            goal=goal_node,
            occ_map=map_obj,
            no_samples=samples,
            rbit=rbit,
            dim=dim,
            stop_time=stop_time,
        )
    path, path_length = planner.make_plan()

    print(planner.ci, planner.old_ci)
    print(path, path_length)

    if vis:
        output_dir = f"{pwd}/../Output/PyViz/{map_name} - {str(datetime.now())}/"
        os.makedirs(output_dir, exist_ok=True)

        print(log_dir, os.listdir(log_dir))
        sim_edges, sim_costs, sim_paths = read_json(log_dir, max_iter=np.inf)
        inv_map = np.where((occ_map==0)|(occ_map==1), occ_map^1, occ_map)
        draw_tree(sim_edges, sim_costs, sim_paths, start, goal, inv_map)
        plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map_name",
        type=str,
        default="Default.png",
        help="Name of the map file. If none, will default to a 100x100 empty grid. Eg: --map_name Default.png",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Whether or not to save and visualize outputs",
    )
    parser.add_argument("--start", nargs="+", help="Start coordinates. Eg: --start 0 0")
    parser.add_argument("--goal", nargs="+", help="Goal coordinates. Eg: --goal 99 99")
    parser.add_argument(
        "--rbit", type=float, default=10, help="Maximum Edge length. Eg: --rbit 10"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of new samples per iteration. Eg: --samples 50",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimensions of working space. Eg: --dim 2"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Eg: --seed 0",
    )
    parser.add_argument(
        "--stop_time",
        type=int,
        default=60,
        help="When to stop the algorithm. Eg: --stop_time 60",
    )

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    opt = parse_opt()
    print(opt)

    assert len(opt.start) == opt.dim
    assert len(opt.goal) == opt.dim

    main(**vars(opt))
