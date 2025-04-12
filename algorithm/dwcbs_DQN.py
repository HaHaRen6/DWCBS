import random
import time as timer
import heapq
import numpy as np
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from ipdb import set_trace as st 
import pickle
from collections import defaultdict

DEBUG = False
time_step_count = 0
total_time = 0

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
            return True
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
    (time_start, time_end) = time_window
    path1, path2 = normalize_paths(pathA, pathB, time_end)
    length = len(path1)
    time_end = min(time_end, length + time_start - 1)
    for t in range(time_start, time_end + 1):
        time_step_count += 1
        # check for vertex collision
        pos1 = get_location(path1, t)
        pos2 = get_location(path2, t)
        if pos1 == pos2:
            # return the vertex and the timestep causing the collision
            return [pos1], t, 'vertex'
        # check for edge collision
        if t < time_end:
            next_pos1 = get_location(path1, t + 1)
            next_pos2 = get_location(path2, t + 1)
            if pos1 == next_pos2 and pos2 == next_pos1:
                # we return the edge and timestep causing the collision
                return [pos1, next_pos1], t + 1, 'edge'
    return None

def detect_collisions(paths, time_window=(None, None)):
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
    prefix_len = 0
    for i in range(min_len):
        if list1[i] == list2[i]:
            prefix_len += 1
        else:
            break  # 遇到不同元素时终止循环
    return prefix_len

def shift_window(time_window, size):
    return (time_window[1], time_window[1] + size - 1)

def adjust_window(new_begin, size):
    return (new_begin, new_begin + size - 1)

def adjust_time_window(curr_window, action):
    # 动作定义:
    # 0: 保持不变；1: +1；2: -1；3: 翻倍；4: 减半；5: 重置为2
    if action == 0:
        return curr_window
    elif action == 1:
        return curr_window + 1
    elif action == 2:
        return max(curr_window - 1, 2)
    elif action == 3:
        return curr_window * 2
    elif action == 4:
        return max(curr_window // 2, 2)
    elif action == 5:
        return 2
    return curr_window
import heapq
import random
import time as timer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# QNetwork 类：定义一个简单的全连接Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQNAgent 类：包含经验回放、epsilon-greedy 策略、目标网络更新等
class DQNAgent(object):
    def __init__(self, action_space, state_dim, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.action_space = action_space
        self.state_dim = state_dim
        self.action_dim = len(action_space)
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        self.policy_net = QNetwork(state_dim, self.action_dim)
        self.target_net = QNetwork(state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.replay_buffer = []
        self.buffer_size = 10000
        self.batch_size = 32
        self.update_steps = 0
        self.target_update_freq = 100

        # 保存上一次状态和动作，用于训练更新
        self.last_state = None
        self.last_action = None

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action_index = q_values.argmax().item()
            return self.action_space[action_index]

    def store_transition(self, state, action, reward, next_state, done):
        action_idx = self.action_space.index(action)
        self.replay_buffer.append((state, action_idx, reward, next_state, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        global total_time
        time1 = timer.time()
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        total_time += timer.time() - time1

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# DWCBSRLSolver 类：动态时间窗CBS，融合了强化学习（DQN）来调整时间窗大小
class DWCBSRLSolver(object):
    """The high-level search of CBS with Dynamic Window and Reinforcement Learning."""
    def __init__(self, my_map, starts, goals, dwindowmin, dwindowstep, train_flag, max_time=60):
        """
        my_map   - 地图障碍物列表
        starts   - 每个agent的起点 [(x, y), ...]
        goals    - 每个agent的目标位置 [(x, y), ...]
        dwindowmin - 最小时间窗大小
        dwindowstep - 时间窗步长
        train_flag - 是否处于训练模式
        max_time   - 最大运行时间
        """
        self.start_time = 0
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.high_level_time = 0
        self.low_level_time = 0
        global time_step_count
        self.total_conflicts = 0
        self.dwindowmin = dwindowmin
        self.dwindowstep = dwindowstep
        self.max_time = max_time if max_time else float('inf')
        self.train_flag = bool(int(train_flag))
        time_step_count = 0
        self.open_list = []
        self.cont = 0
        self.window_size = 3

        # 定义 RL 中动作空间与状态维度
        self.action_space = [
            "no_change",  # 窗口大小保持不变
            "increase",   # 增加窗口大小 +1
            "decrease",   # 减少窗口大小 -1
            "reset",      # 重置为最小窗口 dwindowmin
            "double",     # 窗口大小翻倍
            "halve"       # 窗口大小减半
        ]
        # 状态维度：(当前窗口大小, 上一次窗口大小, 上上次窗口大小, 当前冲突数, 窗口调整类型)
        self.state_dim = 5

        # 初始化 DQN 代理
        self.rl_agent = DQNAgent(
            action_space=self.action_space,
            state_dim=self.state_dim,
            learning_rate=0.1,
            gamma=0.9,
            epsilon=0.1
        )
        self.rl_time = 0

        if not self.train_flag:
            print("加载 Q 表")
            try:
                self.rl_agent.load("q_table.pkl")
            except FileNotFoundError:
                print("Q表文件不存在，将重新训练")

        # 记录窗口大小历史（初始值设为3）
        self.window_size_history = [3, 3]  # [上上次, 上次]
        self.last_conflict_count = 0

        # 计算低层搜索启发式函数（假定 compute_heuristics 函数已实现）
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), 
                                          node['time_window'][0]-node['time_window'][1],
                                          -node['time_window'][1], self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def calculate_reward(self, curr_conflict_count, window_size, solved=False, window_adjustment_type=0):
        """
        计算强化学习奖励
         - solved: 表示是否已找到解，若找到则给予高奖励
         - window_adjustment_type: 0 表示窗口移动，1 表示窗口调整
        """
        # if solved:
        #     return 100

        conflict_reward = -10 * curr_conflict_count
        window_size_penalty = 0.1 * window_size
        stability_reward = 0
        small_window_penalty = -5 if window_size < 3 and curr_conflict_count > 2 else 0
        adjustment_type_reward = 2 if window_adjustment_type == 1 and curr_conflict_count < self.last_conflict_count else (1 if window_adjustment_type == 0 and curr_conflict_count == 0 else 0)

        return conflict_reward + window_size_penalty + stability_reward + small_window_penalty + adjustment_type_reward

    def execute_action(self, action, current_window_size):
        """
        根据动作选择调整后的窗口大小
        """
        return 3
        if action == "no_change":
            return current_window_size
        elif action == "increase":
            return current_window_size + 1
        elif action == "decrease":
            return max(3, current_window_size - 1)
        elif action == "reset":
            return self.dwindowmin
        elif action == "double":
            return current_window_size * 2
        elif action == "halve":
            return max(3, current_window_size // 2)
        return current_window_size

    def find_solution(self):
        """
        寻找所有智能体从起点到目标的路径
        """
        global time_step_count
        time_step_count = 0
        self.start_time = timer.time()

        # 生成根节点
        root = {
            'cost': 0,
            'constraints': [],
            'paths': [],
            'collisions': [],
            'time_window': (None, None)
        }
        if not self.train_flag:
            self.rl_agent.load("q_table.pkl")

        for i in range(self.num_of_agents):
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            self.low_level_time = timer.time() - self.start_time
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        # 初始化时间窗为 [0, window_size-1]
        root['time_window'] = (0, self.window_size - 1)
        root['collisions'] = detect_collisions(root['paths'], root['time_window'])
        self.last_conflict_count = len(root['collisions'])
        self.push_node(root)
        # windowsizefile = open("windowsizerl.csv", "w", buffering=1)

        while self.open_list and timer.time() - self.start_time < self.max_time:
            if self.train_flag:
                self.rl_agent.update()
            p = self.pop_node()
            # 若当前无冲突
            if not p['collisions']:
                if self.all_agents_reached_goals(p['paths'], p['time_window'][1]):
                    current_window_size = p['time_window'][1] - p['time_window'][0] + 1
                    window_adjustment_type = 0  # 窗口移动
                    current_state = (current_window_size,
                                     self.window_size_history[-1],
                                     self.window_size_history[-2],
                                     0,  # 无冲突
                                     window_adjustment_type)
                    reward = self.calculate_reward(0, current_window_size, solved=True, window_adjustment_type=window_adjustment_type)
                    if self.rl_agent.last_state is not None and self.rl_agent.last_action is not None and self.train_flag:
                        self.rl_agent.store_transition(self.rl_agent.last_state,
                                                       self.rl_agent.last_action,
                                                       reward,
                                                       current_state,
                                                       done=True)
                    # windowsizefile.close()
                    time_step = self.print_results(p)
                    if self.train_flag:
                        self.rl_agent.save("q_table.pkl")
                    return p['paths'], time_step, 0, 0
                else:
                    # 当前窗口无冲突时，使用RL选择动作调整时间窗
                    current_window_size = p['time_window'][1] - p['time_window'][0] + 1
                    window_adjustment_type = 0  # 窗口移动类型
                    time33 = timer.time()
                    current_state = (current_window_size,
                                     self.window_size_history[-1],
                                     self.window_size_history[-2],
                                     0,
                                     window_adjustment_type)
                    action = self.rl_agent.select_action(current_state)
                    new_window_size = self.execute_action(action, current_window_size)
                    # windowsizefile.write(str(new_window_size) + "\n")
                    self.rl_time += timer.time() - time33
                    self.window_size_history.append(current_window_size)
                    if len(self.window_size_history) > 3:
                        self.window_size_history.pop(0)

                    # 滑动时间窗并更新冲突检测
                    p['time_window'] = shift_window(p['time_window'], new_window_size)
                    p['collisions'] = detect_collisions(p['paths'], p['time_window'])
                    reward = self.calculate_reward(len(p['collisions']), new_window_size, window_adjustment_type=window_adjustment_type)
                    next_state = (new_window_size,
                                  self.window_size_history[-1],
                                  self.window_size_history[-2],
                                  len(p['collisions']),
                                  window_adjustment_type)
                    if self.rl_agent.last_state is not None and self.rl_agent.last_action is not None and self.train_flag:
                        self.rl_agent.store_transition(self.rl_agent.last_state,
                                                       self.rl_agent.last_action,
                                                       reward,
                                                       next_state,
                                                       done=False)
                    self.rl_agent.last_state = current_state
                    self.rl_agent.last_action = action
                    self.last_conflict_count = len(p['collisions'])
                    p['cost'] = get_sum_of_cost(p['paths'])
                    self.push_node(p)
                    continue

            # 当存在冲突时，取最早冲突进行标准拆分
            self.total_conflicts += len(p['collisions'])
            collision = min(p['collisions'], key=lambda x: x['timestep'])
            constraints = standard_splitting(collision, p['time_window'])
            for c in constraints:
                q = {
                    'cost': 0,
                    'constraints': [*p['constraints'], c],
                    'paths': p['paths'].copy(),
                    'collisions': [],
                    'time_window': p['time_window']
                }
                agent = c['agent']
                tmp_time = timer.time()
                path = a_star(self.my_map, self.starts[agent], self.goals[agent],
                              self.heuristics[agent], agent, q['constraints'])
                self.low_level_time += timer.time() - tmp_time

                if path:
                    new_time_begin = longest_common_prefix(path, q['paths'][agent])
                    q['paths'][agent] = path

                    # 采用 RL 调整窗口大小
                    current_window_size = q['time_window'][1] - q['time_window'][0] + 1
                    window_adjustment_type = 1  # 窗口调整类型
                    time22 = timer.time()
                    current_state = (current_window_size,
                                     self.window_size_history[-1],
                                     self.window_size_history[-2],
                                     len(p['collisions']),
                                     window_adjustment_type)
                    action = self.rl_agent.select_action(current_state)
                    new_window_size = self.execute_action(action, self.dwindowmin)
                    self.rl_time += timer.time() - time22
                    # windowsizefile.write(str(new_window_size) + "\n")

                    self.window_size_history.append(current_window_size)
                    if len(self.window_size_history) > 3:
                        self.window_size_history.pop(0)

                    q['time_window'] = adjust_window(new_time_begin - 1, new_window_size)
                    q['collisions'] = detect_collisions(q['paths'], q['time_window'])
                    reward = self.calculate_reward(len(q['collisions']), new_window_size, window_adjustment_type=window_adjustment_type)
                    next_state = (new_window_size,
                                  self.window_size_history[-1],
                                  self.window_size_history[-2],
                                  len(q['collisions']),
                                  window_adjustment_type)
                    if self.rl_agent.last_state is not None and self.rl_agent.last_action is not None:
                        self.rl_agent.store_transition(self.rl_agent.last_state,
                                                       self.rl_agent.last_action,
                                                       reward,
                                                       next_state,
                                                       done=False)
                    self.rl_agent.last_state = current_state
                    self.rl_agent.last_action = action
                    self.last_conflict_count = len(q['collisions'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)

        if timer.time() - self.start_time >= self.max_time:
            self.CPU_time = self.max_time
            self.high_level_time = self.CPU_time - self.low_level_time
            print("Timeout!")
            return [], -1, 1, 0

        if not self.open_list:
            self.CPU_time = timer.time() - self.start_time
            self.high_level_time = self.CPU_time - self.low_level_time
            print("No solutions!")
            return [], time_step_count, 0, 1

    def all_agents_reached_goals(self, paths, t):
        """ 检查所有智能体是否在时刻 t 达到目标 """
        for i in range(self.num_of_agents):
            if get_location(paths[i], t) != self.goals[i]:
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
        print("Low level time (s):  {:.5f}".format(self.low_level_time))
        print("Sum of costs:        {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:      {}".format(self.num_of_expanded))
        print("Generated nodes:     {}".format(self.num_of_generated))
        print("Detect time steps    {}".format(time_step_count))
        print("Average time steps   {:.5f}".format(time_step_count / self.num_of_expanded))
        print("Total conflicts:     {}".format(self.total_conflicts))
        print("Average conflicts:   {:.5f}".format(self.total_conflicts / self.num_of_expanded))
        print("total_time: {:.5f}".format(total_time))
        print("RL time: {:.5f}".format(self.rl_time))
        collisions = detect_collisions(node['paths'], (0, max_len - 1))
        if collisions == []:
            print("No collisions")
        else:
            raise BaseException('Collisions')
        return time_step_count
