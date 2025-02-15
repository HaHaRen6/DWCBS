import random
import time as timer
import heapq
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from ipdb import set_trace as st 

DEBUG = True
window_size = 5

def normalize_paths(pathA, pathB):
    """
    扩展短的路径，使得两个路径长度相等
    """
    path1 = pathA.copy()
    path2 = pathB.copy()
    shortest, pad = (path1, len(path2) - len(path1)) if len(path1) < len(path2) else (path2, len(path1) - len(path2))
    for _ in range(pad):
        shortest.append(shortest[-1])
    return path1, path2

def detect_collision(pathA, pathB, time_window=(None, None)):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    # 
    # 用于检测一个智能体是否与另一个智能体发生碰撞，即使其中一方已达成目标。
    
    path1, path2 = normalize_paths(pathA, pathB)
    length = len(path1)
    (time_start, time_end) = time_window
    time_start = time_start if time_start is not None else 0
    time_end = time_end if time_end is not None else length
    for t in range(time_start, time_end):
        # check for vertex collision
        pos1 = get_location(path1, t)
        pos2 = get_location(path2, t)
        if pos1 == pos2:
            # return the vertex and the timestep causing the collision
            return [pos1], t, 'vertex'
        # check for edge collision (edge collision 不会发生在最后一步，为啥...)
        if t < length - 1:
            next_pos1 = get_location(path1, t + 1)
            next_pos2 = get_location(path2, t + 1)
            if pos1 == next_pos2 and pos2 == next_pos1:
                # we return the edge and timestep causing the collision
                return [pos1, next_pos1], t + 1, 'edge'
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

def shift_window(time_window, shift1, shift2):
    return (time_window[0] + shift1, time_window[1] + shift2)

class WCBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals, max_time=None):
        """
        my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.start_time = 0
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.max_time =  max_time if max_time else float('inf')

        self.open_list = []
        self.cont = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        if DEBUG:
            print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        if DEBUG:
            print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ 
        Finds paths for all agents from their start locations to their goal locations
        """
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
            'time_window': (None, None)
        }
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        # init time window
        root['time_window'] = (0, window_size)
        root['collisions'] = detect_collisions(root['paths'], root['time_window'])
        self.push_node(root)
        # Task 3.1: Testing
        if DEBUG:
            print("3.1", root['collisions'])
        # Task 3.2: Testing
        if DEBUG:
            for collision in root['collisions']:
                print("3.2", standard_splitting(collision))

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
            # CBS: if there are no collisions, we found a solution
            # WCBS: if there are no collisions and all agents have reached their goals, we found a solution
            if not p['collisions']:
                if self.all_agents_reached_goals(p['paths'], p['time_window'][1]):
                    # find a solution
                    self.print_results(p)
                    return p['paths']
                else:
                    # 当前时间窗无冲突，滑动时间窗
                    p['time_window'] = shift_window(p['time_window'], window_size, window_size)
                    p['collisions'] = detect_collisions(p['paths'], p['time_window'])
                    p['cost'] = get_sum_of_cost(p['paths'])
                    # 将滑动时间窗后的节点加入open_list
                    self.push_node(p)
                    continue
            # 选择第一个冲突
            collision = min(p['collisions'], key=lambda x: x['timestep'])
            constraints = standard_splitting(collision, p['time_window'])
            for c in constraints:
                q = {'cost': 0,
                     'constraints': [*p['constraints'], c],  # all constraints in p plus c
                     'paths': p['paths'].copy(),
                     'collisions': [],
                     'time_window': p['time_window']
                     }
                agent = c['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent],
                              agent, q['constraints'])
                new_time_begin = longest_common_prefix(path, q['paths'][agent])
                if path:
                    q['paths'][agent] = path
                    q['time_window'] = (new_time_begin - 1, new_time_begin + window_size - 1)
                    q['collisions'] = detect_collisions(q['paths'], q['time_window'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)
                else:
                    raise BaseException('No solutions')
        raise BaseException('Time limit exceeded')
    
    def all_agents_reached_goals(self, paths, t):
        """ Check if all agents have reached their goals """
        for i in range(self.num_of_agents):
            if get_location(paths[i], t) != self.goals[i]:  # Check if the last position of agent i is the goal
                return False
        return True

    def print_results(self, node):
        # if DEBUG:
            print("\n Found a solution! \n")
            CPU_time = timer.time() - self.start_time
            print("CPU time (s):    {:.5f}".format(CPU_time))
            print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
            print("Expanded nodes:  {}".format(self.num_of_expanded))
            print("Generated nodes: {}".format(self.num_of_generated))
