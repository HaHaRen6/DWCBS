#!/usr/bin/python
import argparse
import glob
from pathlib import Path
from algorithm.cbs import CBSSolver
from algorithm.wcbs import WCBSSolver
from algorithm.dwcbs_rule import DWCBSSolver
from algorithm.dwcbs_q_learning import DWCBSRLSolver
from algorithm.dwcbs import DWCBSSolver
from algorithm.dwcbs_fast import DWCBSfastSolver
from independent import IndependentSolver
from prioritized import PrioritizedPlanningSolver
from random_instance import random_map, save_map, correct_random_map
from visualize import Animation
from single_agent_planner import get_sum_of_cost
import os
import time as timer
import random
import time
import sys

SOLVER = "WCBS"


def print_mapf_instance(my_map, starts, goals):
    print('Start locations')
    print_locations(my_map, starts)
    print('Goal locations')
    print_locations(my_map, goals)


def print_locations(my_map, locations):
    starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    for i in range(len(locations)):
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            elif my_map[x][y]:
                to_print += '@ '
            else:
                to_print += '. '
        to_print += '\n'
    print(to_print)


def import_mapf_instance(filename, agent_num=20):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    # #agents
    line = f.readline()
    num_agents = min(int(line), agent_num)
    # #agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    return my_map, starts, goals
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('--instance', type=str, default=None,
                        help='The name of the instance file(s)')
    parser.add_argument('--random', action='store_true', default=False,
                        help='Use a random map with auto-genereted agents (see function random_map)')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Runs on benchmark mode (random, success)')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use batch output instead of animation')
    parser.add_argument('--solver', type=str, default=SOLVER,
                        help='The solver to use (one of: {CBS,Independent,Prioritized}), defaults to ' + str(SOLVER))
    parser.add_argument('--windowsize', type=int, default=3,
                        help='window size')
    parser.add_argument('--agentnum', type=int, default=None,
                        help='num of agents')
    # parser.add_argument('--dwindowmin', type=int, default=2,
    #                     help='')
    # parser.add_argument('--dwindowstep', type=int, default=1,
    #                     help='')
    parser.add_argument('--train', default=False)

    args = parser.parse_args()

    if args.benchmark:
        # Benchmark mode
        if args.benchmark == "random":
            map_size = 10;obstacles_dist = .05;max_agents=30
            experiment = 0;max_time = 2*60
            result = {};samples = 25
            start_agents = 4
            for agents in range(start_agents, max_agents + 1,2):
                result[agents] = {
                    'cbs': {'cpu_time':[-1]*samples, 'expanded':[-1]*samples, 'time_steps':[-1]*samples},
                    'wcbs': {'cpu_time':[-1]*samples, 'expanded':[-1]*samples, 'time_steps':[-1]*samples},
                    # 'cbs_disjoint': {'cpu_time':[-1]*samples, 'expanded':[-1]*samples},
                }
                for _ in range(samples):
                    print("Samples {} with {} agents".format(_, agents))
                    my_map, starts, goals = random_map(map_size, map_size, agents, obstacles_dist)
                    filename = "benchmark/max_agents_{}/test_{}.txt".format(agents, _)
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    save_map(my_map, starts, goals, filename)
                    # for alg in ['cbs','wcbs']:
                    #     solver =  CBSSolver(my_map,starts,goals,max_time)
                    #     try:
                    #         solver.find_solution(alg=='cbs_disjoint')
                    #         result[agents][alg]['cpu_time'][_] = round(timer.time() - solver.start_time,2)
                    #     except BaseException as e:
                    #         # Timeout
                    #         pass
                    #     result[agents][alg]['expanded'][_] = solver.num_of_expanded
                    solver1 = CBSSolver(my_map, starts, goals, max_time)
                    path1, time_steps, timeout, nosolution = solver1.find_solution()
                    result[agents]['cbs']['expanded'][_] = solver1.num_of_expanded
                    result[agents]['cbs']['cpu_time'][_] = round(solver1.CPU_time,2)
                    result[agents]['cbs']['time_steps'][_]  = time_steps

                    solver2 = WCBSSolver(my_map, starts, goals, max_time)
                    path2, time_steps, timeout, nosolution = solver2.find_solution()
                    result[agents]['wcbs']['expanded'][_] = solver2.num_of_expanded
                    result[agents]['wcbs']['cpu_time'][_] = round(solver2.CPU_time,2)
                    result[agents]['wcbs']['time_steps'][_]  = time_steps

            with open('benchmark/result.json', 'w') as outfile:
                json.dump(result, outfile)
        if args.benchmark == "success":
            obstacles_dist = .05; map_size = 20; max_agents = 20
            samples = 25
            time_limit = 5*60
            result = {}
            map, starts, goals = random_map(map_size, map_size, max_agents, obstacles_dist)
            save_map(map, starts, goals, "benchmark/{}_agents_success.txt".format(max_agents))
            for agents in range(4,max_agents + 1,2):
                result[agents] = {
                    'cbs': {'cpu_time':[-1]*samples, 'expanded':[-1]*samples, 'time_steps':[-1]*samples},
                    'wcbs': {'cpu_time':[-1]*samples, 'expanded':[-1]*samples, 'time_steps':[-1]*samples},
                    # 'cbs_disjoint': {'cpu_time':[-1]*samples, 'expanded':[-1]*samples},
                }
                for i in range(samples):
                    # take first i agents
                    random.shuffle(starts);sub_goals = goals[0:agents]
                    random.shuffle(goals);sub_starts = starts[0:agents]
                    print("sample {} with {} agents".format(i,agents))
                    # for alg in ['cbs','cbs_disjoint']:
                    #     solver = CBSSolver(map,sub_starts,sub_goals,time_limit)
                    #     try:
                    #         solver.find_solution(alg=='cbs_disjoint')
                    #         result[agents][alg]['cpu_time'][i] = round(timer.time() - solver.start_time, 2)
                    #     except BaseException as e:
                    #         # Timeout
                    #         pass
                    #     result[agents][alg]['expanded'][i] = solver.num_of_expanded
                    print("Run CBS")
                    solver1 = CBSSolver(map, sub_starts, sub_goals, time_limit)
                    path1, time_steps = solver1.find_solution()
                    result[agents]['cbs']['expanded'][i] = solver1.num_of_expanded
                    result[agents]['cbs']['cpu_time'][i] = round(solver1.CPU_time,2)
                    result[agents]['cbs']['time_steps'][i]  = time_steps

                    print("Run WCBS")
                    solver2 = WCBSSolver(map, sub_starts, sub_goals, time_limit)
                    path2, time_steps = solver2.find_solution()
                    result[agents]['wcbs']['expanded'][i] = solver2.num_of_expanded
                    result[agents]['wcbs']['cpu_time'][i] = round(solver2.CPU_time,2)
                    result[agents]['wcbs']['time_steps'][i]  = time_steps
            print(result)
            with open('benchmark/result_success.json', 'w') as outfile:
                json.dump(result, outfile)

    else:
        # Otherwise, run the algorithm
        files = ["random.generated"] if args.random else glob.glob(args.instance, recursive=True)
        for file in files:
            print("***Import an instance*** " + file)
            max_agent_num = int(args.agentnum) if args.agentnum else 10000
            my_map, starts, goals = random_map(8, 8, 6, .1) if args.random else import_mapf_instance(file, max_agent_num) # 可以限制agent num
            print_mapf_instance(my_map, starts, goals)
            save_map(my_map, starts, goals, 'img/output_map.txt')
            if args.solver == "CBS":
                print("***Run CBS***")
                solver = CBSSolver(my_map, starts, goals)
                paths, time_steps, timeout, nosolution = solver.find_solution()
            elif args.solver == "WCBS":
                print("***Run WCBS***")
                solver = WCBSSolver(my_map, starts, goals, window_size=args.windowsize)
                paths, time_steps, timeout, nosolution = solver.find_solution()
            # elif args.solver == "DWCBS_rule":
            #     print("***Run DWCBS_rule***")
            #     solver = DWCBSSolver(my_map, starts, goals, args.dwindowmin, args.dwindowstep)
            #     paths, time_steps, timeout, nosolution = solver.find_solution() 
            # elif args.solver == "DWCBSRL":
            #     print("***Run DWCBSRL***")
            #     solver = DWCBSRLSolver(my_map, starts, goals, args.train)
            #     paths, time_steps, timeout, nosolution = solver.find_solution() 
            elif args.solver == "DWCBS":
                print("***Run DWCBS***")
                solver = DWCBSSolver(my_map, starts, goals, args.train)
                paths, time_steps, timeout, nosolution = solver.find_solution() 
            elif args.solver == "DWCBSfast":
                print("***Run DWCBS***")
                solver = DWCBSfastSolver(my_map, starts, goals)
                paths, time_steps, timeout, nosolution = solver.find_solution() 
            elif args.solver == "Independent":
                print("***Run Independent***")
                solver = IndependentSolver(my_map, starts, goals)
                paths = solver.find_solution()
            elif args.solver == "Prioritized":
                print("***Run Prioritized***")
                solver = PrioritizedPlanningSolver(my_map, starts, goals)
                paths = solver.find_solution()
            elif args.solver == "map_only":
                paths = []
                animation = Animation(my_map, [], [], [])
                # animation.save_video("output1.gif", 1.0)
                animation.save_static_images("./output/" + args.instance.rsplit('/', 2)[-2].split('.', 1)[0])
                animation.show()
                sys.exit(0)
            else:
                raise RuntimeError("Unknown solver!")

            cost = get_sum_of_cost(paths)
            num_of_agents = solver.num_of_agents
            basename0 = os.path.basename(os.path.dirname(file))
            basename = os.path.basename(os.path.dirname(os.path.dirname(file)))
            # print(basename)
            if basename and basename != '.':
                csvname = "results/results-" + basename + "-" + basename0 + ".csv"
                # csvname = "results/results-" + os.path.basename(os.path.dirname(file)) +args.solver + "-" + str(args.windowsize) + str(args.dwindowmin) +str(args.dwindowstep) + ".csv"
                # csvname = "results/results-" + os.path.basename(os.path.dirname(file)) + args.solver + "-" + str(args.windowsize) + ".csv"
                # csvname = "results/results-" + os.path.basename(os.path.dirname(file)) + ".csv"
            else:
                csvname = "results/results-others.csv"
            result_file = open(csvname, "a", buffering=1)
            # print(result_file)

            average_conflicts = solver.total_conflicts / solver.num_of_expanded

            if args.solver == "WCBS":
                result_file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(os.path.basename(file).split('.')[0], args.solver + "-" + str(args.windowsize), num_of_agents, cost, solver.CPU_time, solver.high_level_time, time_steps, timeout, nosolution, average_conflicts, time.ctime()))
            elif args.solver == "DWCBS" or args.solver == "DWCBSfast":
                # result_file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(os.path.basename(file).split('.')[0], args.solver + "-" + str(args.dwindowmin)+ "-" + str(args.dwindowstep), num_of_agents, cost, solver.CPU_time, solver.high_level_time, time_steps, timeout, nosolution, average_conflicts, time.ctime()))
                result_file.write("{},{},{},{},{},{},{},{},{}\n".format(os.path.basename(file).split('.')[0], "DWCBS", num_of_agents, cost, solver.CPU_time, solver.high_level_time, time_steps, timeout, nosolution, average_conflicts, time.ctime()))
            elif args.solver == "Independent":  
                result_file.write("{},{},{},{},{},{},{}\n".format(os.path.basename(file).split('.')[0], args.solver, num_of_agents, cost, solver.CPU_time, 0, 0, time.ctime()))
            elif args.solver != "map_only":
                result_file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(os.path.basename(file).split('.')[0], args.solver, num_of_agents, cost, solver.CPU_time, solver.high_level_time, time_steps, timeout, nosolution, average_conflicts, time.ctime()))

            # if not args.batch:
                # print("***Test paths on a simulation***")
                # animation = Animation(my_map, starts, goals, paths)
                # animation.save_video("output1.gif", 1.0)
                # animation.save_static_images("./output/" + args.instance.rsplit('/', 1)[-1].split('.', 1)[0])
                # animation.show()
    print("***Done***")
    result_file.close()
