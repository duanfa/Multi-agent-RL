import random
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grid_env

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # macOS使用Arial Unicode MS，Windows使用SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置图片保存目录
save_dir = "plot_figure/policy_iteration_iterations"

class  class_policy_iteration:
    def __init__(self,env: grid_env.GridEnv):
        self.gama = 0.9  # discount rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2  # 幂运算，grid world的尺寸 如 5 ** 2 = 25的网格世界。
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list  # 父类中：self.reward_list = [0, 1, -10, -10]
        # state_value
        self.state_value = np.zeros(shape=self.state_space_size)  # 1维数组
        # action value -> Q-table
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 25 x 5

        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("./logs")  # 实例化SummaryWriter对象

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("state_value.shape:{} , qvalue.shape:{} , mean_p olicy.shape:{}".format(self.state_value.shape,
                                                                                     self.qvalue.shape,
                                                                                     self.mean_policy.shape))
        print("\n分别是non-forbidden area, target area, forbidden area 以及撞墙:")
        print("self.reward_space_size:{},self.reward_list:{}".format(self.reward_space_size, self.reward_list))
        print('----------------------------------------------------------------')

    def random_greed_policy(self):
        """
        生成随机的greedy策略
        :return:
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state_index in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))  #从数组中随机抽取元素
            policy[state_index, action] = 1  # 选中该动作作为策略，即置1
        print("random_choice_policy",policy)
        return policy

    def policy_iteration(self, tolerance=0.001, steps=100, save_images=False):
        """

            :param tolerance: 迭代前后policy的范数小于tolerance 则认为已经收敛
            :param steps: step 小的时候就退化成了  truncated iteration
            :param save_images: 是否保存每次迭代的图片
            :return: 剩余迭代次数
        """
        # 初始化保存目录
        if save_images:
            if os.path.exists(save_dir):
                for file in os.listdir(save_dir):
                    os.remove(os.path.join(save_dir, file))
            else:
                os.makedirs(save_dir)
        
        iteration_count = 0
        policy = self.random_greed_policy()
        while np.linalg.norm(policy - self.policy, ord=1) > tolerance and steps > 0:
            steps -= 1
            policy = self.policy.copy()
            self.state_value = self.policy_evaluation(self.policy.copy(), tolerance, steps)
            self.policy, _ = self.policy_improvement(self.state_value)  #只接收第一个返回值 （更关注第一个返回值
            
            # 保存当前迭代的图片
            if save_images:
                self.save_iteration_image(iteration_count, self.state_value, self.qvalue, self.policy)
            
            iteration_count += 1
            
        return steps

    def policy_evaluation(self, policy, tolerance=0.001, steps=10):
        """
        迭代求解贝尔曼公式 得到 state value tolerance 和 steps 满足其一即可
        :param policy: 需要求解的policy
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止计算 此时若是policy iteration 则算法变为 truncated iteration
        :return: 求解之后的收敛值
        """
        state_value_k = np.ones(self.state_space_size)
        state_value = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k - state_value, ord=1) > tolerance:  # While j < jtruncate, do
            state_value = state_value_k.copy()
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):
                    value += policy[state, action] * self.calculate_qvalue(state_value=state_value_k.copy(),
                                                                           state=state,
                                                                           action=action)  # bootstrapping
                state_value_k[state] = value
        return state_value_k

    def calculate_qvalue(self, state, action, state_value):
        """
        计算qvalue elementwise形式
        :param state: 对应的state
        :param action: 对应的action
        :param state_value: 状态值
        :return: 计算出的结果
        """
        qvalue = 0
        for i in range(self.reward_space_size):
            qvalue += self.reward_list[i] * self.env.Rsa[state, action, i]
        for next_state in range(self.state_space_size):
            qvalue += self.gama * self.env.Psa[state, action, next_state] * state_value[next_state]
        return qvalue

    def policy_improvement(self, state_value):
        """
        是普通 policy_improvement 的变种 相当于是值迭代算法 也可以 供策略迭代使用 做策略迭代时不需要 接收第二个返回值
        更新 qvalue ；qvalue[state,action]=reward+value[next_state]
        找到 state 处的 action*：action* = arg max(qvalue[state,action]) 即最优action即最大qvalue对应的action
        更新 policy ：将 action*的概率设为1 其他action的概率设为0 这是一个greedy policy
        :param: state_value: policy对应的state value
        :return: improved policy, 以及迭代下一步的state_value
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        state_value_k = state_value.copy()
        q_table = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state in range(self.state_space_size):
            qvalue_list = []
            for action in range(self.action_space_size):
                qvalue_list.append(self.calculate_qvalue(state, action, state_value.copy()))
            q_table[state, :] = qvalue_list.copy()
            state_value_k[state] = max(qvalue_list)
            action_star = qvalue_list.index(max(qvalue_list))
            policy[state, action_star] = 1
        self.qvalue = q_table  # 保存 Q-table
        return policy, state_value_k



    def save_iteration_image(self, iteration, state_value, q_table, policy):
        """
        保存当前迭代的可视化图片
        :param iteration: 当前迭代次数
        :param state_value: 当前状态值
        :param q_table: 当前Q值表
        :param policy: 当前策略
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建新图形
        grid_size = int(np.sqrt(self.state_space_size))
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 绘制网格
        for i in range(grid_size + 1):
            ax.axhline(i, color='black', linewidth=1.5)
            ax.axvline(i, color='black', linewidth=1.5)
        
        # 标记禁止区域和目标区域
        for state in range(self.state_space_size):
            pos = self.env.state2pos(state)
            row, col = pos[1], pos[0]  # 注意：pos是(x,y)，需要转换为(row,col)
            
            # 判断是否是禁止区域
            is_forbidden = False
            for forbidden_pos in self.env.forbidden_location:
                if np.array_equal(pos, forbidden_pos):
                    is_forbidden = True
                    break
            
            # 判断是否是目标区域
            is_target = np.array_equal(pos, self.env.target_location)
            
            # 填充背景色
            if is_forbidden:
                # 禁止区域：橙色
                ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor='orange', alpha=0.6))
            elif is_target:
                # 目标区域：青色
                ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor='cyan', alpha=0.6))
        
        # 绘制策略箭头、状态值和各方向Q值
        for state in range(self.state_space_size):
            pos = self.env.state2pos(state)
            row, col = pos[1], pos[0]
            center_x = col + 0.5
            center_y = row + 0.5
            
            # 检查是否是目标区域
            is_target = np.array_equal(pos, self.env.target_location)
            
            if is_target:
                # 目标区域画圆圈
                circle = plt.Circle((center_x, center_y), 0.15, color='green', fill=False, linewidth=2)
                ax.add_patch(circle)
            
            # 绘制策略箭头（目标区域也绘制）
            for action in range(self.action_space_size):
                policy_prob = policy[state, action]
                if policy_prob > 0:
                    direction = self.env.action_to_direction[action]
                    arrow_length = 0.2
                    dx = direction[0] * arrow_length * policy_prob
                    dy = direction[1] * arrow_length * policy_prob
                    
                    if abs(dx) > 0.01 or abs(dy) > 0.01:  # 只绘制有意义的箭头
                        ax.arrow(center_x, center_y, dx, dy, 
                               head_width=0.08, head_length=0.06, 
                               fc='green', ec='green', linewidth=1.5)
                    elif action == 4 and policy_prob > 0:  # 停留动作，绘制一个实心圆点
                        circle_stay = plt.Circle((center_x, center_y), 0.08, color='green', fill=False)
                        ax.add_patch(circle_stay)
            
            # 显示各个方向的Q值
            # 动作顺序：上(0), 右(1), 下(2), 左(3), 停留(4)
            action_positions = {
                0: (center_x, center_y - 0.35),      # 上
                1: (center_x + 0.35, center_y),      # 右
                2: (center_x, center_y + 0.35),      # 下
                3: (center_x - 0.35, center_y),      # 左
                4: (center_x, center_y)              # 停留/中心
            }
            
            action_names = {
                0: '↑',
                1: '→',
                2: '↓',
                3: '←',
                4: '○'
            }
            
            # 显示每个动作的Q值
            for action in range(self.action_space_size):
                if action < len(action_positions):
                    q_value = q_table[state, action]
                    pos_x, pos_y = action_positions[action]
                    action_name = action_names.get(action, str(action))
                    
                    # 显示动作符号和Q值
                    if action == 4:  # 中心位置（停留动作）
                        # 在中心显示状态值
                        value_text = f"V:{state_value[state]:.2f}"
                        ax.text(pos_x, pos_y - 0.05, value_text, 
                               ha='center', va='center', fontsize=7, fontweight='bold', color='blue')
                        # 停留动作的Q值显示在下方
                        q_text = f"{action_name}:{q_value:.2f}"
                        ax.text(pos_x, pos_y + 0.08, q_text, 
                               ha='center', va='center', fontsize=6, color='purple')
                    else:
                        # 其他方向显示Q值
                        q_text = f"{action_name}:{q_value:.2f}"
                        ax.text(pos_x, pos_y, q_text, 
                               ha='center', va='center', fontsize=7, color='red')
        
        # 设置坐标轴
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 翻转y轴使(0,0)在左上角
        
        # 设置刻度标签
        ax.set_xticks(np.arange(grid_size) + 0.5)
        ax.set_yticks(np.arange(grid_size) + 0.5)
        ax.set_xticklabels(range(1, grid_size + 1))
        ax.set_yticklabels(range(1, grid_size + 1))
        
        # 将刻度移到顶部
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        plt.title(f'策略迭代 - 第 {iteration} 次迭代 (显示各方向Q值)', fontsize=14, pad=20)
        
        # 保存图片
        save_path = os.path.join(save_dir, f'iteration_{iteration:03d}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存第 {iteration} 次迭代图片: {save_path}")

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def obtain_episode(self, policy, start_state, start_action, length):
        """

        :param policy: 由指定策略产生episode
        :param start_state: 起始state
        :param start_action: 起始action
        :param length: episode 长度
        :return: 一个 state,action,reward,next_state,next_action 序列
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = (self.env.step+
                                     (action))
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode


if __name__ == "__main__":
    print("-----Begin!-----")
    # reward_list[other, target, forbidden, overflow]
    gird_world2x2 = grid_env.GridEnv(size=3, target=[2, 2],
                           forbidden=[[1, 0],[2,1]],
                           render_mode='',reward_list=[0, 100, -400, -10])
    solver = class_policy_iteration(gird_world2x2)
    start_time = time.time()

    demand_step = 10000
    # 启用图片保存功能
    remaining_steps = demand_step - solver.policy_iteration(tolerance=0.001, steps=demand_step, save_images=True)
    if remaining_steps > 0:
        print("Policy iteration converged in {} steps.".format(demand_step - remaining_steps))
    else:
        print("Policy iteration did not converge in {} steps.".format(demand_step))

    end_time = time.time()

    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    print(len(gird_world2x2.render_.trajectory))
    print("solver.policy:{}".format(solver.policy))
    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)

    gird_world2x2.render()