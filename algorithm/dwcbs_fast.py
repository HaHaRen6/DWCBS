import random
import time as timer
import heapq
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from ipdb import set_trace as st 
from algorithm.sarsa_agent import SARSA_Agent
import pickle
from collections import deque, defaultdict
import copy

# WRITE_WINDOW_SIZE = False
# WRITE_NORM = False
# DEBUG = False
Q_TABLE_PATH = "q_table/q_table.pkl"
time_step_count = 0
def find_common_duplicate_indices(lst, window):
    if not lst:
        return []
    
    # 计算所有子列表允许的最大索引 i（确保所有子列表至少有 i+2 个元素）
    max_i_per_sublist = [len(sublist) - 2 for sublist in lst]
    overall_max_i = min(max_i_per_sublist)
    
    # 如果存在子列表长度不足，直接返回空
    if overall_max_i < 0:
        return []
    
    common_indices = []
    for i in range(window[0], min(window[1], overall_max_i + 1)):
        # 检查所有子列表在 i 和 i+1 位置是否重复
        all_duplicate = True
        for sublist in lst:
            if sublist[i] != sublist[i + 1]:
                all_duplicate = False
                break
        if all_duplicate:
            # print(i, lst)
            return True
            # common_indices.append(i)
    return False

def normalize_paths(pathA, pathB, time_window_size):
    """
    扩展短的路径，使得两个路径长度相等，长度不超过时间窗长度
    """
    path1 = pathA.copy()[:time_window_size + 1]
    path2 = pathB.copy()[:time_window_size + 1]
    shortest, pad = (path1, len(path2) - len(path1)) if len(path1) < len(path2) else (path2, len(path1) - len(path2))
    for _ in range(pad):
        shortest.append(shortest[-1])
    return path1, path2

def detect_collision(pathA, pathB, time_window):
    global time_step_count
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    # 
    # 用于检测一个智能体是否与另一个智能体发生碰撞，即使其中一方已达成目标。
    (time_start, time_end) = time_window
    path1, path2 = normalize_paths(pathA, pathB, time_end)
    length = len(path1)
    time_end = min(time_end, length + time_start - 1)
    for t in range(time_start, time_end + 1):
        # check for vertex collision
        pos1 = get_location(path1, t)
        pos2 = get_location(path2, t)
        if pos1 == pos2:
            # return the vertex and the timestep causing the collision
            time_step_count += t - time_start + 1
            return [pos1], t, 'vertex'
        # check for edge collision
        if t < time_end:
            next_pos1 = get_location(path1, t + 1)
            next_pos2 = get_location(path2, t + 1)
            if pos1 == next_pos2 and pos2 == next_pos1:
                # we return the edge and timestep causing the collision
                time_step_count += t - time_start + 1
                return [pos1, next_pos1], t + 1, 'edge'
    time_step_count += time_end - time_start + 1
    return None

def detect_collisions(paths, time_window=(None, None)):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    
    collisions = []
    # i and j are agents
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            data = detect_collision(paths[i], paths[j], time_window)
            # if data is not None (collision detected)
            if data:
                collisions.append({
                    'a1': i,
                    'a2': j,
                    'loc': data[0],  # vertex or edge
                    'timestep': data[1],  # timestep
                    'type': data[2]
                })
    return collisions

def standard_splitting(collision, time_window=None):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    #
    # 忽略final参数（normalize_paths()函数已解决）
    
    constraints = []
    if collision['type'] == 'vertex':
        constraints.append({
            'agent': collision['a1'],
            'loc': collision['loc'],
            'timestep': collision['timestep'],
            'final': False,
            'time_window': time_window
        })
        constraints.append({
            'agent': collision['a2'],
            'loc': collision['loc'],
            'timestep': collision['timestep'],
            'final': False,
            'time_window': time_window
        })
    elif collision['type'] == 'edge':
        constraints.append({
            'agent': collision['a1'],
            'loc': collision['loc'],
            'timestep': collision['timestep'],
            'final': False,
            'time_window': time_window
        })
        constraints.append({
            'agent': collision['a2'],
            'loc': list(reversed(collision['loc'])),
            'timestep': collision['timestep'],
            'final': False,
            'time_window': time_window
        })
    return constraints

def longest_common_prefix(list1, list2):
    # 处理空列表的情况
    if not list1 or not list2:
        return []
    min_len = min(len(list1), len(list2))
    # prefix = []
    prefix_len = 0
    for i in range(min_len):
        if list1[i] == list2[i]:
            # prefix.append(list1[i])
            prefix_len += 1
        else:
            break  # 遇到不同元素时终止循环
    return prefix_len

def shift_window(time_window, size):
    return (time_window[1], time_window[1] + size - 1)

def adjust_window(new_begin, size):
    return (new_begin, new_begin + size - 1)

class DWCBSfastSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals, max_time=30):
        """
        my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.start_time = 0
        self.my_map = my_map
        self.map_scale = int((len(my_map) * len(my_map[0])) ** 0.5)
        self.starts = starts
        self.goals = goals
        self.init_window_size = max(2, int(0.3 * self.map_scale))
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.high_level_time = 0
        self.low_level_time = 0
        self.max_time =  max_time if max_time else float('inf')

        self.open_list = []
        self.cont = 0
        global time_step_count
        self.total_conflicts = 0
        time_step_count = 0
        
        self.rltime = 0
        self.action_space = [0, +0.1, -0.1, +0.2, -0.2, -3]
        self.rl_agent = SARSA_Agent(self.action_space)

        # if not self.train_flag:

        self.rl_agent.load(Q_TABLE_PATH)

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']),self.num_of_generated, node))
        # if DEBUG:
        #     print("Generate node {}".format(self.num_of_generated))
        #     # print(node)
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # if DEBUG:
        #     print("Expand node {}".format(id))
        #     # print(node)
        self.num_of_expanded += 1
        return node


    def find_solution(self):
        """ 
        Finds paths for all agents from their start locations to their goal locations
        """
        #TODO time_step bug
        # global time_step_count
        # time_step_count = 0
        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {
            'cost': 0,
            'constraints': [],
            'paths': [],
            'collisions': [],
            'time_window': (None, None),
            'last_window_op': 0,
        }
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            self.low_level_time = timer.time() - self.start_time
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        # init time window
        root['time_window'] = (0, self.init_window_size - 1) # 时间窗为左闭右闭区间
        root['collisions'] = detect_collisions(root['paths'], root['time_window'])
        root['last_window_op'] = 0
        self.total_conflicts = len(root['collisions'])
        self.push_node(root)
        # Task 3.1: Testing
        # if DEBUG:
        #     print("3.1", root['collisions'])
        # # Task 3.2: Testing
        # if DEBUG:
        #     for collision in root['collisions']:
        # #         print("3.2", standard_splitting(collision, root['time_window']))
        # if WRITE_WINDOW_SIZE:
        #     windowsizefile = open("windowsizerl.csv", "w", buffering=1)
        

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node())
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
        while self.open_list and timer.time() - self.start_time < self.max_time:
            p = self.pop_node()
            # print(p)
            # 查看cost到哪了
            # if DEBUG:
            #     if p['cost'] != last_cost:
            #         print("cost {} time {:.5f} expand {} gen {}".format(p['cost'], timer.time() - self.start_time, self.num_of_expanded, self.num_of_generated))
            #     last_cost = p['cost']

            # CBS: if there are no collisions, we found a solution
            # WCBS: if there are no collisions and all agents have reached their goals, we found a solution
            if not p['collisions']:
                if self.all_agents_reached_goals(p['paths'], p['time_window'][1]):
                    # find a solution!!
                    time_step = self.print_results(p)
                    # print(RLAgent.compute_q_table_distance(self.rl_agent.q_table, self.old_q_table))
                    # print(self.rl_agent.q_table)
                    # print(self.old_q_table)
                    
                    # L1, L2 = SARSA_Agent.compute_q_table_distance(self.rl_agent.q_table, self.old_q_table)
                    # print("L1: " + str(L1) + "\tL2: " + str(L2))
                    # if WRITE_NORM:
                    #     normfile = open("norm.csv", "a", buffering=1)
                    #     normfile.write(str(L1) + "," + str(L2) + "\n")
                    #     normfile.close()
                    # self.rl_agent.save(Q_TABLE_PATH)
                    return p['paths'], time_step, 0, 0
                else:
                    # 当前时间窗无冲突，滑动时间窗
                    
                    time11 = timer.time()
                    current_window_size = p['time_window'][1] - p['time_window'][0] + 1
                    # state
                    current_state = self.rl_agent.discretize_state(
                        p['last_window_op'],
                        current_window_size / self.map_scale,
                        0, # 冲突数，此时为0
                        self.total_conflicts / self.num_of_expanded # aver_con
                    )
                    # 根据state选择action
                    action = self.rl_agent.select_action(current_state)
                    new_window_size = self.init_window_size if action == -3 else max(current_window_size + int(action * self.map_scale), 2)
                    self.rltime += timer.time() - time11
                    # if WRITE_WINDOW_SIZE:
                    #     windowsizefile.write(str(round(new_window_size / self.map_scale, 2)) + "\n")

                    p['time_window'] = shift_window(p['time_window'], new_window_size)
                    p['collisions'] = detect_collisions(p['paths'], p['time_window'])
                    p['cost'] = get_sum_of_cost(p['paths'])
                    p['last_window_op'] = 0

                    # if self.train_flag:
                    #     time11 = timer.time()
                    #     # 计算reward
                    #     reward = (
                    #         - len (p['collisions']) * self.rl_agent.a1
                    #         - abs(action) * self.rl_agent.a2
                    #         + new_window_size / self.map_scale * self.rl_agent.a3
                    #     )
                    #     new_state = self.rl_agent.discretize_state(
                    #         0, # 时间窗移动
                    #         new_window_size,
                    #         len (p['collisions']), # 冲突数
                    #         (self.total_conflicts + len(p['collisions'])) / (self.num_of_expanded + 1) # aver_con
                    #     )
                    
                    #     self.rl_agent.update_q_table(current_state, action, reward, new_state)
                    #     self.rltime += timer.time() - time11

                    # 将滑动时间窗后的节点加入open_list
                    self.push_node(p)
                    continue
            # print(p['collisions'])
            self.total_conflicts += len(p['collisions'])
            collision = min(p['collisions'], key=lambda x: x['timestep'])
            constraints = standard_splitting(collision, p['time_window'])
            for c in constraints:
                q = {'cost': 0,
                     'constraints': [*p['constraints'], c],  # all constraints in p plus c
                     'paths': p['paths'].copy(),
                     'collisions': [],
                     'time_window': p['time_window'],
                     'last_window_op': p['last_window_op'],
                     }
                agent = c['agent']
                tmp_time = timer.time()
                path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent],
                              agent, q['constraints'])
                self.low_level_time += timer.time() - tmp_time
                if path:
                    new_time_begin = longest_common_prefix(path, q['paths'][agent])
                    q['paths'][agent] = path
                    
                    time11 = timer.time()
                    current_window_size = q['time_window'][1] - q['time_window'][0] + 1
                    current_state = self.rl_agent.discretize_state(
                        p['last_window_op'],
                        current_window_size / self.map_scale,
                        len(p['collisions']), # 冲突数
                        self.total_conflicts / self.num_of_expanded # aver_con
                    )
                    action = self.rl_agent.select_action(current_state) 
                    new_window_size = self.init_window_size if action == -3 else max(current_window_size + int(action * self.map_scale), 2)
                    self.rltime += timer.time() - time11

                    # if WRITE_WINDOW_SIZE:
                    #     windowsizefile.write(str(round(new_window_size / self.map_scale, 2)) + "\n")
                    
                    q['time_window'] = adjust_window(new_time_begin - 1, new_window_size)
                    q['collisions'] = detect_collisions(q['paths'], q['time_window'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    q['last_window_op'] = 1

                    # if self.train_flag:
                    #     time11 = timer.time()
                    #     reward = (
                    #         - len (p['collisions']) * self.rl_agent.a1
                    #         - abs(action) * self.rl_agent.a2
                    #         + new_window_size / self.map_scale * self.rl_agent.a3
                    #     )
                    #     new_state = self.rl_agent.discretize_state(
                    #         1, # 时间窗调整
                    #         new_window_size / self.map_scale,
                    #         len(p['collisions']), # 冲突数
                    #         (self.total_conflicts + len(p['collisions'])) / (self.num_of_expanded + 1) # aver_con
                    #     )
                    #     self.rl_agent.update_q_table(current_state, action, reward, new_state)
                    #     self.rltime += timer.time() - time11

                    self.push_node(q)
                # else:
                #     raise BaseException('No solutions')
        # raise BaseException('Time limit exceeded')
        if timer.time() - self.start_time >= self.max_time:
            self.CPU_time = self.max_time
            self.high_level_time = self.CPU_time - self.low_level_time
            print("Timeout!")
            # print(RLAgent.compute_q_table_distance(self.rl_agent.q_table, self.old_q_table))
            # L1, L2 = SARSA_Agent.compute_q_table_distance(self.rl_agent.q_table, self.old_q_table)
            # print("L1: " + str(L1) + "\tL2: " + str(L2))
            # if WRITE_NORM:
            #     normfile = open("norm.csv", "a", buffering=1)
            #     normfile.write(str(L1) + "," + str(L2) + "\n")
            #     normfile.close()
            # self.rl_agent.save(Q_TABLE_PATH)
            return [], -1, 1, 0
        
        if not self.open_list:
            # raise BaseException('No solutions')
            self.CPU_time = timer.time() - self.start_time
            self.high_level_time = self.CPU_time - self.low_level_time
            print("No solutions!")
            return [], time_step_count, 0, 1
        
    
    def all_agents_reached_goals(self, paths, t):
        """ Check if all agents have reached their goals """
        for i in range(self.num_of_agents):
            if get_location(paths[i], t) != self.goals[i]:  # Check if the last position of agent i is the goal
                return False
        return True

    def print_results(self, node):
        print("\nFound a solution!")
        self.CPU_time = timer.time() - self.start_time
        self.high_level_time = self.CPU_time - self.low_level_time
        max_len = 0
        for i in range(len(node["paths"])):
            print("agent", i, ": ", node["paths"][i])
            max_len = max(max_len, len(node["paths"][i]))
        print("\nCPU time (s):        {:.5f}".format(self.CPU_time))
        print("High level time (s): {:.5f}".format(self.high_level_time))
        print(" RL time             {:.5f}".format(self.rltime))
        print("Low level time (s):  {:.5f}".format(self.low_level_time))
        print("Sum of costs:        {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:      {}".format(self.num_of_expanded))
        print("Generated nodes:     {}".format(self.num_of_generated))
        print("Detect time steps    {}".format(time_step_count))
        print("Average time steps   {}".format(time_step_count / self.num_of_expanded))
        print("Total conflicts:     {}".format(self.total_conflicts))
        print("Average conflicts:   {}".format(self.total_conflicts / self.num_of_expanded))
        print("Total RL reward:     {:.2f}".format(self.rl_agent.total_reward))

        time_step = time_step_count # 后续检测时还会增加
        # print(max_len)
        collisions = detect_collisions(node['paths'], (0, max_len - 1))
        if collisions == []:
            print("No collisions")
        else:
            raise BaseException('Collisions')
        return time_step
