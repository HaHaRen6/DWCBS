import random
import pickle
from collections import defaultdict
import numpy as np

class RLAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        action_space: 可选动作列表，如 [0, +1, -1, +2, -2, 3]，
                 其中 0 表示时间窗不变
        alpha: 学习率
        gamma: 折扣因子
        epsilon: 探索率
        """
        self.q_table = {}  # Q-table，键为状态(tuple)，值为各动作的 Q 值字典
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.total_reward = 0
        self.a1 = 1 # - len (p['collisions']) 冲突惩罚
        self.a2 = 0.2 # - abs(action)           动作稳定
        self.a3 = 0.1 # + new_window_size       大窗口奖励

    def discretize_state(self, adjust_mode, window_size, collision_count, avg_collision):
        """
        离散化状态：
        
        离散值           0       1       2       3       4
        adjust_mode     shift   adjust
        window_size     2~4     5~7     8~10    11+
        collision_count 0       1~2     3~5     6~9     10+
        avg_collision   0~0.5   0.5~1.5 1.5~3   3~7     7+
        """

        # 时间窗大小
        if 2 <= window_size <= 4:
            ws_bucket = 0
        elif 5 <= window_size <= 7:
            ws_bucket = 1
        elif 8 <= window_size <= 11:
            ws_bucket = 2
        else:
            ws_bucket = 3

        # 冲突数
        if collision_count == 0:
            coll_bucket = 0
        elif 1 <= collision_count <= 2:
            coll_bucket = 1
        elif 3 <= collision_count <= 5:
            coll_bucket = 2
        elif 6 <= collision_count <= 9:
            coll_bucket = 3
        else:
            coll_bucket = 4

        # 离散化平均冲突数
        if 0 <= avg_collision <= 0.5:
            avg_bucket = 0
        elif 0.5 < avg_collision <= 1.5:
            avg_bucket = 1
        elif 1.5 < avg_collision <= 3:
            avg_bucket = 2
        elif 3 < avg_collision <= 7:
            avg_bucket = 3
        else:
            avg_bucket = 4

        return (adjust_mode, ws_bucket, coll_bucket, avg_bucket)

    def get_q(self, state, action):
        """
        若状态 action 尚未存在于 Q-table 中，则初始化为 0
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.action_space}
        return self.q_table[state].get(action,0.0)

    def select_action(self, state):
        """
            epsilon-greedy
        """
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        
        # 如果状态没有出现过，则先初始化
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.action_space}

        q_values = self.q_table[state]
        max_q = max(q_values.values())
        # 如果多个动作Q相同，则随机选一个
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)


    def update_q_table(self, state, action, reward, next_state):
        """
        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.action_space}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.action_space}
        max_next_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.total_reward += reward

    def print_q_table(self):
        print("当前Q表内容：")
        # 对状态进行排序
        sorted_states = sorted(self.q_table.keys())
        for state in sorted_states:
            actions = self.q_table[state]
            formatted_state = ", ".join(map(str, state))
            # 计算动作的最大宽度
            max_action_width = max(len(str(a)) for a in actions.keys())
            # 计算Q值的最大宽度
            max_q_width = max(len(f"{q:.2f}") for q in actions.values())
            formatted_actions = ", ".join(f"\t{a:>{max_action_width}}: {q:>{max_q_width}.2f}" for a, q in actions.items())
            print(f"状态 ({formatted_state}): {formatted_actions},\t {max(actions, key=actions.get)}")
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    # @staticmethod
    def load(self, filename):
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)
            # print(loaded_dict)
            default_q = lambda: [0.0, 0.0, 0.0, 0.0]
            self.q_table = defaultdict(default_q, loaded_dict)

    @staticmethod
    def compute_q_table_distance(q1, q2):
        """
        计算两个 Q 表之间的 L1 和 L2 范数。

        参数：
            q1, q2: dict，格式为 {state: {action: value}}

        返回：
            l1_norm: float，L1 范数
            l2_norm: float，L2 范数
        """
        # print(q1)
        
        all_keys = set(q1.keys()) | set(q2.keys())
        diffs = []
        # print(all_keys)

        for state in all_keys:
            actions1 = q1.get(state, {})
            actions2 = q2.get(state, {})
            all_actions = set(actions1.keys()) | set(actions2.keys())

            for action in all_actions:
                v1 = actions1.get(action, 0.0)
                v2 = actions2.get(action, 0.0)
                diffs.append(v1 - v2)

        diffs = np.array(diffs)
        l1_norm = np.linalg.norm(diffs, ord=1)
        l2_norm = np.linalg.norm(diffs, ord=2)

        return round(l1_norm, 2), round(l2_norm, 2)
