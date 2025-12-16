import numpy as np
import platform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

# ===================== 修复中文乱码 - 关键配置 =====================
plt.rcParams['font.sans-serif'] = [
    # Windows系统
    # 'Microsoft YaHei', 'SimHei', 'DejaVu Sans',
    # Mac系统
    'Heiti TC', 'PingFang SC', 'Hiragino Sans GB',
    # Linux系统
    # 'WenQuanYi Micro Hei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 额外兼容：自动检测系统并设置字体（可选，增强鲁棒性）
def set_chinese_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    elif system == "Darwin":  # Mac
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang SC']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

# 执行字体配置
set_chinese_font()

# ===================== 全局配置（与文档设定一致）=====================
GRID_SIZE = 5  # 5x5网格
FORBIDDEN_STATE = (1, 1)  # 禁止区域（行1,列1，注意：代码中索引从0开始，实际对应文档(2,2)）
TARGET_STATE = (4, 4)     # 目标区域（行4,列4，对应文档(5,5)）
REWARD_BOUNDARY = -1      # 边界奖励
REWARD_FORBIDDEN = -1     # 禁止区域奖励
REWARD_OTHER = -1         # 非特殊区域奖励
REWARD_TARGET = 0         # 目标奖励
GAMMA = 0.9               # 折扣因子
ALPHA = 0.1               # 学习率
EPSILON = 0.1             # Sarsa/Q-learning的ε-贪婪策略参数
EPISODES = 100            # 迭代总回合数
SAVE_DIR = "images"       # 图片保存目录

# 创建保存目录
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 定义动作：上、右、下、左、停留（对应坐标变化）
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
ACTION_NAMES = ["上", "右", "下", "左", "停留"]

# 自定义颜色映射（状态值/动作值可视化）
cmap_v = LinearSegmentedColormap.from_list("state_value", ["#FFE4E1", "#FFFFFF", "#E0FFFF"])  # 红→白→蓝
cmap_q = LinearSegmentedColormap.from_list("action_value", ["#FF6B6B", "#4ECDC4", "#45B7D1"])  # 红→青→蓝

# ===================== 工具函数（网格绘制、状态转换等）=====================
def is_boundary(state):
    """判断是否为边界状态"""
    i, j = state
    return i < 0 or i >= GRID_SIZE or j < 0 or j >= GRID_SIZE

def is_forbidden(state):
    """判断是否为禁止区域"""
    return state == FORBIDDEN_STATE

def is_target(state):
    """判断是否为目标区域"""
    return state == TARGET_STATE

def get_reward(state, next_state):
    """根据当前状态和下一状态获取奖励"""
    if is_target(state):
        return REWARD_TARGET
    if is_forbidden(state) or is_boundary(next_state):
        return REWARD_FORBIDDEN if is_forbidden(state) else REWARD_BOUNDARY
    return REWARD_OTHER

def transition(state, action):
    """状态转换：根据当前状态和动作获取下一状态"""
    if is_target(state):
        return state  # 目标状态终止，不转换
    i, j = state
    ni, nj = i + action[0], j + action[1]
    next_state = (ni, nj)
    # 边界或禁止区域：停留原地
    if is_boundary(next_state) or is_forbidden(next_state):
        return state
    return next_state

def draw_state_value_grid(v, episode, title="状态值网格"):
    """绘制状态值网格图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(v, cmap=cmap_v, vmin=-2, vmax=0.5)  # 适配状态值范围

    # 绘制网格线
    for i in range(GRID_SIZE + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)

    # 标注禁止区域和目标区域
    fi, fj = FORBIDDEN_STATE
    ax.add_patch(patches.Rectangle((fj - 0.5, fi - 0.5), 1, 1, facecolor='orange', alpha=0.8))
    ti, tj = TARGET_STATE
    ax.add_patch(patches.Rectangle((tj - 0.5, ti - 0.5), 1, 1, facecolor='blue', alpha=0.8))

    # 标注状态值
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) == FORBIDDEN_STATE:
                ax.text(j, i, "禁止", ha='center', va='center', fontsize=12, fontweight='bold')
            elif (i, j) == TARGET_STATE:
                ax.text(j, i, "目标", ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                ax.text(j, i, f"{v[i, j]:.2f}", ha='center', va='center', fontsize=10)

    # 设置坐标轴
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels([f"列{j+1}" for j in range(GRID_SIZE)])
    ax.set_yticklabels([f"行{i+1}" for i in range(GRID_SIZE)])
    ax.set_title(f"{title}（迭代{episode}）", fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("状态值 $v(s)$", fontsize=12)

    # 保存图片
    save_path = os.path.join(SAVE_DIR, f"iteration{episode}_state_value.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"状态值图已保存：{save_path}")

def draw_action_value_grid(q, episode, algorithm="Sarsa", title="动作值网格"):
    """绘制动作值网格图（标注每个状态的最优动作）"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 自动检测网格大小（从传入的q数组获取）
    actual_grid_size = q.shape[0]
    
    # 计算每个状态的最大动作值（用于可视化强度）
    max_q = np.max(q, axis=2)
    im = ax.imshow(max_q, cmap=cmap_q, vmin=-2, vmax=1)

    # 绘制网格线
    for i in range(actual_grid_size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)

    # 标注禁止区域和目标区域（仅当它们在实际网格范围内时）
    fi, fj = FORBIDDEN_STATE
    if fi < actual_grid_size and fj < actual_grid_size:
        ax.add_patch(patches.Rectangle((fj - 0.5, fi - 0.5), 1, 1, facecolor='orange', alpha=0.8))
    ti, tj = TARGET_STATE
    if ti < actual_grid_size and tj < actual_grid_size:
        ax.add_patch(patches.Rectangle((tj - 0.5, ti - 0.5), 1, 1, facecolor='blue', alpha=0.8))

    # 标注每个状态的最优动作和动作值
    for i in range(actual_grid_size):
        for j in range(actual_grid_size):
            if (i, j) == FORBIDDEN_STATE:
                ax.text(j, i, "禁止", ha='center', va='center', fontsize=12, fontweight='bold')
            elif (i, j) == TARGET_STATE:
                ax.text(j, i, "目标", ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                # 最优动作（最大Q值对应的动作）
                best_action_idx = np.argmax(q[i, j])
                best_action = ACTION_NAMES[best_action_idx]
                best_q = q[i, j, best_action_idx]
                # 标注最优动作和Q值
                ax.text(j, i-0.1, best_action, ha='center', va='center', fontsize=11, fontweight='bold')
                ax.text(j, i+0.1, f"{best_q:.2f}", ha='center', va='center', fontsize=9)

    # 设置坐标轴
    ax.set_xlim(-0.5, actual_grid_size - 0.5)
    ax.set_ylim(-0.5, actual_grid_size - 0.5)
    ax.set_xticks(range(actual_grid_size))
    ax.set_yticks(range(actual_grid_size))
    ax.set_xticklabels([f"列{j+1}" for j in range(actual_grid_size)])
    ax.set_yticklabels([f"行{i+1}" for i in range(actual_grid_size)])
    ax.set_title(f"{algorithm} {title}（迭代{episode}）", fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("最大动作值 $max_a q(s,a)$", fontsize=12)

    # 保存图片
    save_path = os.path.join(SAVE_DIR, f"iteration{episode}_{algorithm}_action_value.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"动作值图已保存：{save_path}")

def draw_convergence_curve(rewards, lengths, algorithm="Sarsa", title="迭代曲线"):
    """绘制迭代曲线（总奖励+回合长度）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 总奖励曲线
    ax1.plot(range(len(rewards)), rewards, color='#1f77b4', linewidth=1.5, label='总奖励')
    ax1.set_ylabel("每回合总奖励", fontsize=12)
    ax1.set_title(f"{algorithm} {title}", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 回合长度曲线
    ax2.plot(range(len(lengths)), lengths, color='#ff7f0e', linewidth=1.5, label='回合长度')
    ax2.set_xlabel("迭代回合数", fontsize=12)
    ax2.set_ylabel("每回合步数", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 保存图片
    save_path = os.path.join(SAVE_DIR, f"{algorithm}_convergence_curve.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"迭代曲线图已保存：{save_path}")

def draw_algorithm_comparison(rewards_dict, lengths_dict, title="算法对比"):
    """绘制多算法对比曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红
    algorithms = list(rewards_dict.keys())

    # 总奖励对比
    for i, algo in enumerate(algorithms):
        ax1.plot(range(len(rewards_dict[algo])), rewards_dict[algo], 
                 color=colors[i], linewidth=1.5, label=algo)
    ax1.set_ylabel("每回合总奖励", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 回合长度对比
    for i, algo in enumerate(algorithms):
        ax2.plot(range(len(lengths_dict[algo])), lengths_dict[algo], 
                 color=colors[i], linewidth=1.5, label=algo)
    ax2.set_xlabel("迭代回合数", fontsize=12)
    ax2.set_ylabel("每回合步数", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 保存图片
    save_path = os.path.join(SAVE_DIR, "algorithm_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"算法对比图已保存：{save_path}")

# ===================== 7.1 状态值的TD学习 =====================
def td_state_value_learning():
    """TD状态值学习（对应7.1节）"""
    print("\n=== 开始生成7.1节：状态值的TD学习可视化 ===")
    # 初始化状态值（所有状态初始为0）
    v = np.zeros((GRID_SIZE, GRID_SIZE))
    # 记录每回合的总奖励和长度（用于收敛曲线）
    rewards_history = []
    lengths_history = []

    # 绘制初始状态（迭代0）
    draw_state_value_grid(v, episode=0, title="TD状态值学习-初始状态")

    for episode in range(EPISODES):
        # 随机初始化起始状态（避开禁止区域和目标区域）
        while True:
            start_i = np.random.randint(GRID_SIZE)
            start_j = np.random.randint(GRID_SIZE)
            current_state = (start_i, start_j)
            if not is_forbidden(current_state) and not is_target(current_state):
                break

        total_reward = 0
        step = 0
        current_s = current_state

        # 单回合迭代
        while not is_target(current_s):
            step += 1
            # 随机策略：等概率选择4个动作
            action_idx = np.random.choice(len(ACTIONS))
            action = ACTIONS[action_idx]
            # 状态转换
            next_s = transition(current_s, action)
            # 获取奖励
            reward = get_reward(current_s, next_s)
            total_reward += reward
            # TD状态值更新公式：v(s) = v(s) + α[r + γ*v(s') - v(s)]
            i, j = current_s
            ni, nj = next_s
            v[i, j] += ALPHA * (reward + GAMMA * v[ni, nj] - v[i, j])
            # 转移到下一状态
            current_s = next_s

            # 防止步数过长
            if step > 1000:
                break

        # 记录历史数据
        rewards_history.append(total_reward)
        lengths_history.append(step)

        # 关键迭代节点保存图片（对应文档中的迭代1、5、10、50）
        if episode in [0, 4, 9, 49]:  # 代码中episode从0开始，对应文档迭代1、5、10、50
            draw_state_value_grid(v, episode=episode+1, title="TD状态值学习")

    # 绘制最终收敛状态（迭代100）
    draw_state_value_grid(v, episode=EPISODES, title="TD状态值学习-最终收敛")
    # 绘制收敛曲线
    draw_convergence_curve(rewards_history, lengths_history, algorithm="TD-State-Value", title="状态值学习收敛曲线")
    print("=== 7.1节可视化生成完成 ===")
    return v

# ===================== 7.2 Sarsa算法（动作值学习）=====================
def sarsa_learning():
    """Sarsa算法（对应7.2节）"""
    print("\n=== 开始生成7.2节：Sarsa算法可视化 ===")
    # 初始化动作值（5x5网格 x 4个动作）
    q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    rewards_history = []
    lengths_history = []

    # 绘制初始动作值网格（迭代0）
    draw_action_value_grid(q, episode=0, algorithm="Sarsa", title="动作值学习-初始状态")

    def epsilon_greedy(state):
        """ε-贪婪策略选择动作"""
        if np.random.random() < EPSILON:
            return np.random.choice(len(ACTIONS))  # 随机选择
        else:
            i, j = state
            return np.argmax(q[i, j])  # 最优动作

    for episode in range(EPISODES):
        # 随机初始化起始状态
        while True:
            start_i = np.random.randint(GRID_SIZE)
            start_j = np.random.randint(GRID_SIZE)
            current_state = (start_i, start_j)
            if not is_forbidden(current_state) and not is_target(current_state):
                break

        total_reward = 0
        step = 0
        current_s = current_state
        current_a = epsilon_greedy(current_s)  # 初始动作

        while not is_target(current_s):
            step += 1
            # 状态转换
            action = ACTIONS[current_a]
            next_s = transition(current_s, action)
            # 获取奖励
            reward = get_reward(current_s, next_s)
            total_reward += reward
            # 选择下一动作（Sarsa是同策略，用当前策略选择）
            next_a = epsilon_greedy(next_s)
            # Sarsa更新公式：q(s,a) = q(s,a) + α[r + γ*q(s',a') - q(s,a)]
            i, j = current_s
            ni, nj = next_s
            q[i, j, current_a] += ALPHA * (reward + GAMMA * q[ni, nj, next_a] - q[i, j, current_a])
            # 转移到下一状态和动作
            current_s = next_s
            current_a = next_a

            if step > 1000:
                break

        rewards_history.append(total_reward)
        lengths_history.append(step)

        # 关键迭代节点保存图片（迭代1、10、20、60）
        if episode in [0, 9, 19, 59]:
            draw_action_value_grid(q, episode=episode+1, algorithm="Sarsa", title="动作值学习")

    # 最终收敛状态（迭代100）
    draw_action_value_grid(q, episode=EPISODES, algorithm="Sarsa", title="动作值学习-最终收敛")
    # 绘制收敛曲线
    draw_convergence_curve(rewards_history, lengths_history, algorithm="Sarsa", title="Sarsa算法收敛曲线")
    print("=== 7.2节可视化生成完成 ===")
    return q, rewards_history, lengths_history

# ===================== 7.3 n步Sarsa算法 =====================
def n_step_sarsa_learning(n=3):
    """n步Sarsa算法（对应7.3节，默认n=3）"""
    print(f"\n=== 开始生成7.3节：{n}步Sarsa算法可视化 ===")
    q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    rewards_history = []
    lengths_history = []

    # 绘制初始动作值网格（迭代0）
    draw_action_value_grid(q, episode=0, algorithm=f"{n}-Step-Sarsa", title="动作值学习-初始状态")

    def epsilon_greedy(state):
        """ε-贪婪策略选择动作"""
        if np.random.random() < EPSILON:
            return np.random.choice(len(ACTIONS))
        else:
            i, j = state
            return np.argmax(q[i, j])

    for episode in range(EPISODES):
        # 随机初始化起始状态
        while True:
            start_i = np.random.randint(GRID_SIZE)
            start_j = np.random.randint(GRID_SIZE)
            current_state = (start_i, start_j)
            if not is_forbidden(current_state) and not is_target(current_state):
                break

        # 初始化轨迹缓存（存储s, a, r）
        trajectory = []
        total_reward = 0
        step = 0
        current_s = current_state
        current_a = epsilon_greedy(current_s)
        trajectory.append((current_s, current_a, 0))  # 初始r=0

        while not is_target(current_s):
            step += 1
            # 状态转换
            action = ACTIONS[current_a]
            next_s = transition(current_s, action)
            # 获取奖励
            reward = get_reward(current_s, next_s)
            total_reward += reward
            # 选择下一动作
            next_a = epsilon_greedy(next_s)
            # 加入轨迹缓存
            trajectory.append((next_s, next_a, reward))
            # n步更新：当轨迹长度≥n+1时，更新t时刻的(s,a)
            if len(trajectory) > n:
                t = len(trajectory) - n - 1
                s_t, a_t, _ = trajectory[t]
                s_n, a_n, _ = trajectory[t + n]
                # 计算n步回报 G_t = r_{t+1} + γ*r_{t+2} + ... + γ^n * q(s_n, a_n)
                G = 0
                for k in range(t + 1, t + n + 1):
                    G += (GAMMA ** (k - t - 1)) * trajectory[k][2]
                G += (GAMMA ** n) * q[s_n[0], s_n[1], a_n]
                # 更新动作值
                i, j = s_t
                q[i, j, a_t] += ALPHA * (G - q[i, j, a_t])
            # 转移到下一状态和动作
            current_s = next_s
            current_a = next_a

            if step > 1000:
                break

        # 处理剩余轨迹（不足n步的部分）
        while len(trajectory) > 1:
            t = len(trajectory) - 2
            s_t, a_t, _ = trajectory[t]
            # 计算剩余回报（到目标状态）
            G = 0
            for k in range(t + 1, len(trajectory)):
                G += (GAMMA ** (k - t - 1)) * trajectory[k][2]
            # 更新动作值
            i, j = s_t
            q[i, j, a_t] += ALPHA * (G - q[i, j, a_t])
            trajectory.pop()

        rewards_history.append(total_reward)
        lengths_history.append(step)

        # 关键迭代节点保存图片（迭代1、15、40）
        if episode in [0, 14, 39]:
            draw_action_value_grid(q, episode=episode+1, algorithm=f"{n}-Step-Sarsa", title="动作值学习")

    # 最终收敛状态（迭代100）
    draw_action_value_grid(q, episode=EPISODES, algorithm=f"{n}-Step-Sarsa", title="动作值学习-最终收敛")
    # 绘制收敛曲线
    draw_convergence_curve(rewards_history, lengths_history, algorithm=f"{n}-Step-Sarsa", title=f"{n}步Sarsa算法收敛曲线")
    print(f"=== 7.3节可视化生成完成 ===")
    return q, rewards_history, lengths_history

# ===================== 7.4 Q-学习算法 =====================
def q_learning():
    """Q-学习算法（对应7.4节）"""
    print("\n=== 开始生成7.4节：Q-学习算法可视化 ===")
    q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    rewards_history = []
    lengths_history = []

    # 绘制初始动作值网格（迭代0）
    draw_action_value_grid(q, episode=0, algorithm="Q-Learning", title="动作值学习-初始状态")

    def epsilon_greedy_behavior(state):
        """行为策略（ε-贪婪，高探索性，ε=0.2）"""
        if np.random.random() < 0.2:
            return np.random.choice(len(ACTIONS))
        else:
            i, j = state
            return np.argmax(q[i, j])

    for episode in range(EPISODES):
        # 随机初始化起始状态
        while True:
            start_i = np.random.randint(GRID_SIZE)
            start_j = np.random.randint(GRID_SIZE)
            current_state = (start_i, start_j)
            if not is_forbidden(current_state) and not is_target(current_state):
                break

        total_reward = 0
        step = 0
        current_s = current_state

        while not is_target(current_s):
            step += 1
            # 行为策略选择动作
            current_a = epsilon_greedy_behavior(current_s)
            # 状态转换
            action = ACTIONS[current_a]
            next_s = transition(current_s, action)
            # 获取奖励
            reward = get_reward(current_s, next_s)
            total_reward += reward
            # Q-学习更新公式：q(s,a) = q(s,a) + α[r + γ*max_a q(s',a) - q(s,a)]
            i, j = current_s
            ni, nj = next_s
            max_q_next = np.max(q[ni, nj])  # 目标策略是贪婪策略，取最大Q值
            q[i, j, current_a] += ALPHA * (reward + GAMMA * max_q_next - q[i, j, current_a])
            # 转移到下一状态（无需关注下一动作）
            current_s = next_s

            if step > 1000:
                break

        rewards_history.append(total_reward)
        lengths_history.append(step)

        # 关键迭代节点保存图片（迭代1、10、30）
        if episode in [0, 9, 29]:
            draw_action_value_grid(q, episode=episode+1, algorithm="Q-Learning", title="动作值学习")

    # 最终收敛状态（迭代100）
    draw_action_value_grid(q, episode=EPISODES, algorithm="Q-Learning", title="动作值学习-最终收敛")
    # 绘制收敛曲线
    draw_convergence_curve(rewards_history, lengths_history, algorithm="Q-Learning", title="Q-学习算法收敛曲线")
    print("=== 7.4节可视化生成完成 ===")
    return q, rewards_history, lengths_history

# ===================== 7.5 算法对比 =====================
def algorithm_comparison(sarsa_rewards, sarsa_lengths, nstep_rewards, nstep_lengths, q_rewards, q_lengths):
    """算法对比（对应7.5节）"""
    print("\n=== 开始生成7.5节：算法对比可视化 ===")
    # 构建对比数据字典
    rewards_dict = {
        "Sarsa": sarsa_rewards,
        f"{3}-Step-Sarsa": nstep_rewards,
        "Q-Learning": q_rewards
    }
    lengths_dict = {
        "Sarsa": sarsa_lengths,
        f"{3}-Step-Sarsa": nstep_lengths,
        "Q-Learning": q_lengths
    }
    # 绘制对比曲线
    draw_algorithm_comparison(rewards_dict, lengths_dict, title="TD算法收敛性能对比")
    print("=== 7.5节可视化生成完成 ===")

# ===================== 主函数（执行所有可视化生成）=====================
if __name__ == "__main__":
    # 7.1 状态值的TD学习
    # td_v = td_state_value_learning()

    # 7.2 Sarsa算法
    # sarsa_q, sarsa_rewards, sarsa_lengths = sarsa_learning()

    # # 7.3 3步Sarsa算法
    # nstep_q, nstep_rewards, nstep_lengths = n_step_sarsa_learning(n=3)

    # # 7.4 Q-学习算法
    # q_q, q_rewards, q_lengths = q_learning()

    # # 7.5 算法对比
    # algorithm_comparison(sarsa_rewards, sarsa_lengths, nstep_rewards, nstep_lengths, q_rewards, q_lengths)

    print("\n" + "="*50)
    print("所有可视化图片已生成完成！")
    print(f"图片保存路径：{os.path.abspath(SAVE_DIR)}")
    print("="*50)
