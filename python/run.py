import argparse
from bit_star_vis import *
import bit_star_vis
from vis3 import *
import sys
import shutil


def main(map_name, vis, start, goal, rbit, samples, dim, seed, stop_time, fast):
    pwd = os.path.abspath(os.path.dirname(__file__))


    time_taken_all = []
    path_lengths = []

    text_path = f"{pwd}/../Output/path_lengths_and_times_{map_name}.txt"
    with open(text_path, 'a') as f:
        f.write("Seed,Path Length,Time Taken\n")


    for seed in range(seed):



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

        map_path = f"{pwd}/../gridmaps/{map_name}.png"
        occ_map = np.array(Image.open(map_path))

        bit_star_vis.start_arr = start
        bit_star_vis.goal_arr = goal

        start_node = Node(tuple(start), gt=0)
        goal_node = Node(tuple(goal))

        map_obj = Map(start=start_node, goal=goal_node, occ_grid=occ_map)
        planner = None
        if vis or fast:
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
        path, path_length, time_taken = planner.make_plan()

        print(planner.ci, planner.old_ci)
        print(path, path_length)

        time_taken_all.append(time_taken)
        path_lengths.append(path_length)

        print(f"Seed: {seed}, Path Length: {path_length}, Time Taken: {time_taken}")

        time_taken_str = ','.join([str(t) for t in time_taken])


        with open(text_path, 'a') as f:
            f.write(f"{seed},{path_length},{time_taken_str}\n")


    if vis or fast:
        output_dir = f"{pwd}/../Output/{map_name} - {str(datetime.now())}/"
        os.makedirs(output_dir, exist_ok=True)

        inv_map = np.where((occ_map == 0) | (occ_map == 1), occ_map ^ 1, occ_map)

        visualizer = Visualizer(start, goal, inv_map, output_dir)

        print(log_dir, os.listdir(log_dir))
        visualizer.read_json(log_dir, max_iter=np.inf)

        for i in range(len(os.listdir(log_dir))):
            visualizer.draw(i, fast)
        visualizer.ax.set_title("BIT* - Final Path", fontsize=30)
        print("Done drawing")
        plt.show()

    # Delete the log directory
    print("Deleting log directory: ", log_dir)
    shutil.rmtree(log_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map_name",
        type=str,
        default="Default",
        help="Name of the map file without file extension as this only takes '.png'. If none, will default to a 100x100 empty grid. Eg: --map_name Default",
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
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Whether or not to only plot the final edge list of each iteration. Note: when this flag is set the vis is also considered set. Eg: --fast",
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