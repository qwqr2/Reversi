import math
import copy
import numpy as np
import random
from board import Board

'''
采用神经网络改进探索策略的蒙特卡洛树搜索
'''

def softmax(x):
    """将数组转换为概率分布"""
    # 避免数值溢出
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


class Node_plus(object):
    '''
    蒙特卡洛树节点 - 包含神经网络评估信息的增强版
    '''
    def __init__(self):
        # 基础属性
        self.color = None           # 节点代表的玩家颜色
        self.board = Board()        # 节点对应的棋盘状态
        self.candidate = None       # 到达此节点的落子位置
        self.visit = 0              # 访问次数
        self.score = 0              # 得分: 黑棋赢为+1分，平为0分，白棋赢为-1分
        
        # 树结构相关
        self.parent = None          # 父节点
        self.child = []             # 子节点列表
        self.childnodes = []        # 子节点落子位置列表
        self.next_locations = None  # 从当前节点可行的落子位置
        
        # 状态与神经网络相关
        self.status = 0             # 扩展状态: 0=未完全扩展, 1=完全扩展
        self.prob = 1.0             # 节点先验概率
        self.nextlocation_prob = None  # 神经网络预测的各位置先验概率
    

class Mcts_plus(object):
    '''
    神经网络增强的蒙特卡洛树搜索
    '''
    
    def __init__(self, board, policy_value_function, r=400, is_selfplay=0, c_puct=1/math.sqrt(2)):
        """
        初始化MCTS搜索器
        
        参数:
            board: 当前棋盘状态
            policy_value_function: 策略价值网络评估函数
            r: 搜索模拟次数
            is_selfplay: 是否为自我对弈模式(0=否, 1=是)
            c_puct: UCB公式中的探索常数
        """
        self.color = board.color 
        self.board = copy.deepcopy(board)
        self.r = r                        # 迭代次数
        self.func = policy_value_function # 策略价值函数
        self.is_selfplay = is_selfplay    # 自我对弈模式标志
        self.c_puct = c_puct              # UCB探索参数
        
    def ucb1(self, node):
        """
        使用PUCT公式计算节点价值
        PUCT结合了先验概率与UCB公式
        """
        # 如果节点未被访问过，返回一个很大的值以鼓励探索
        if node.visit == 0:
            return float('inf')
            
        # UCB1公式: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploitation = node.score / node.visit  # 利用项
        exploration = self.c_puct * node.prob * math.sqrt(2 * node.parent.visit) / (1 + node.visit)  # 探索项
        return exploitation + exploration
        
    def selection(self, node):
        """
        选择阶段: 根据UCB值选择最有前途的节点
        """
        current_node = node
        # 当节点已完全扩展，继续向下选择
        while current_node.status == 1 and current_node.child:
            best_value = float('-inf')
            best_child = None
            
            # 选择UCB值最大的子节点
            for child_node in current_node.child:
                ucb_value = self.ucb1(child_node)
                if ucb_value > best_value:
                    best_value = ucb_value
                    best_child = child_node
                    
            current_node = best_child
            
        return current_node
            
    def expand(self, node):
        """
        扩展阶段: 创建一个新的子节点
        """
        # 找出所有未扩展的候选位置
        unexpanded_candidates = []
        for pos in node.next_locations:
            if pos not in node.childnodes:
                unexpanded_candidates.append(pos)
                
        # 如果只剩一个未扩展位置，标记为完全扩展
        if len(unexpanded_candidates) == 1:
            node.status = 1
            
        if not unexpanded_candidates:
            return None
        
        # 如果节点没有先验概率，先进行模拟
        if node.nextlocation_prob is None:
            self.simulation(node)
        
        # 选择具有最高先验概率的候选位置
        max_prob = float('-inf')
        best_candidate = None
        
        for candidate in unexpanded_candidates:
            x, y = candidate
            if node.nextlocation_prob[x][y] > max_prob:
                max_prob = node.nextlocation_prob[x][y]
                best_candidate = candidate
                
        # 创建新节点
        new_node = Node_plus()
        
        # 设置新节点颜色
        if node.parent is None:  # 根节点，扩展的子节点颜色相同
            new_node.color = self.color
        else:  # 切换颜色
            new_node.color = 'O' if node.color == 'X' else 'X'
            
        # 设置节点属性
        new_node.prob = max_prob
        new_node.parent = node
        new_node.candidate = best_candidate
        new_node.visit = 1  # 初始化访问次数为1
        new_node.board = copy.deepcopy(node.board)  # 复制棋盘状态
        new_node.board.reversi_pieces(best_candidate)  # 执行落子
        new_node.next_locations = new_node.board.locations()  # 更新可行位置
        
        # 更新父节点
        node.childnodes.append(best_candidate)
        node.child.append(new_node)
        
        return new_node
 
    def simulation(self, node):
        """
        模拟阶段: 使用神经网络评估当前节点
        
        注意: 不再执行随机模拟，而是直接使用神经网络评估
        """
        # 创建棋盘副本避免修改原棋盘
        board_copy = copy.deepcopy(node.board)
        board_copy.locations()  # 计算可行位置
        board_copy.pieces_index()  # 更新棋子计数
        
        # 使用策略价值网络评估当前状态
        # 注意：神经网络视角是当前玩家，返回的score是从对手角度看的
        node.nextlocation_prob, value = self.func(board_copy)
        
        # 将价值取反，转换为当前玩家视角
        node.score = -value
        
    def back_update(self, node):
        """
        反向传播阶段: 更新节点及其祖先节点的统计信息
        """
        current_node = node
        current_score = node.score
        
        # 向上更新所有祖先节点
        while current_node.parent is not None:
            current_node.parent.visit += 1
            current_score = -current_score  # 切换玩家视角
            current_node.parent.score += current_score
            current_node = current_node.parent
            
    def mcts_run(self):
        """
        执行完整的MCTS搜索过程
        
        返回:
            在对战模式下: 返回最佳行动和所有行动的概率分布
            在自我对弈模式下: 返回按照Dirichlet噪声采样的行动和概率分布
        """
        # 创建根节点
        root = Node_plus()
        root.color = self.color
        root.board = self.board
        root.board.color = root.color
        root.next_locations = root.board.locations()
        
        # 初始评估根节点
        self.simulation(root)
        
        # 如果没有合法落子，直接返回
        if not root.next_locations:
            return None, np.zeros((8, 8))
        
        # MCTS主循环
        for _ in range(self.r):
            # 选择阶段
            node = self.selection(root)
            
            # 如果游戏结束，直接评估
            if node.board.is_game_over():
                node.score = node.board.win()
                self.back_update(node)
                continue
                
            # 扩展阶段
            if node.status == 0:
                expand_node = self.expand(node)
                if expand_node:
                    node = expand_node
                    
            # 模拟阶段
            self.simulation(node)
            
            # 反向传播
            self.back_update(node)
            
        # 选择访问次数最多的动作
        best_action = None
        best_visit = -1
        
        for child in root.child:
            if child.visit > best_visit:
                best_visit = child.visit
                best_action = child.candidate
                
        # 获取动作概率分布
        action_probs = np.zeros((8, 8))
        for child in root.child:
            action_probs[child.candidate[0]][child.candidate[1]] = child.visit
            
        # 归一化概率分布
        if np.sum(action_probs) > 0:
            action_probs = action_probs / np.sum(action_probs)
            
        return best_action, action_probs