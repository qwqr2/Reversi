import math
import copy
import random
from board import Board

'''
蒙特卡洛树搜索算法 - 纯MCTS实现版本
优化版本: 提高搜索效率并增强可读性
'''

class Node(object):
    '''
    蒙特卡洛树搜索节点
    '''
    def __init__(self):
        '''
        节点属性初始化
        '''
        # 基础属性
        self.color = None           # 节点代表的玩家颜色
        self.board = Board()        # 节点对应的棋盘状态
        self.coordinate = None      # 到达此节点的落子位置
        self.visit = 0              # 访问次数
        self.score = 0              # 得分: 黑棋赢为+1分，平为0分，白棋赢为-1分
        
        # 树结构相关
        self.parent = None          # 父节点
        self.child = []             # 子节点列表
        self.childnodes = []        # 子节点落子位置列表
        self.next_locations = None  # 从当前节点可行的落子位置
        self.status = 0             # 扩展状态: 0=未完全扩展, 1=完全扩展
    

class Mcts(object):
    '''
    蒙特卡洛树搜索的实现
    '''
    
    def __init__(self, board, r):
        '''
        初始化MCTS搜索器
        
        参数:
            board: 当前棋盘状态
            r: 搜索模拟次数
        '''
        self.color = board.color 
        self.board = copy.deepcopy(board)
        self.r = r    # 迭代次数
        
    def ucb1(self, node, c=1/math.sqrt(2)):
        '''
        计算UCB1值，平衡探索与利用
        
        参数:
            node: 要评估的节点
            c: 探索常数 (1/sqrt(2) 通常是个好选择)
        '''
        # UCB1公式: Q(v)/N(v) + c*sqrt(2*ln(N(p))/N(v))
        # Q(v) = 节点总得分
        # N(v) = 节点访问次数
        # N(p) = 父节点访问次数
        return node.score/node.visit + c*math.sqrt(2*math.log(node.parent.visit)/node.visit)
        
    def selection(self, node):
        '''
        选择阶段: 根据UCB1值选择最有前途的节点
        '''
        selection_node = node
        # 当节点已完全扩展且有子节点时，选择UCB1值最高的子节点
        while selection_node.status == 1 and selection_node.child:
            best_child_value = -float('inf')
            best_child = None
            
            for child in selection_node.child:
                ucb_value = self.ucb1(child)
                if ucb_value > best_child_value:
                    best_child_value = ucb_value
                    best_child = child
                    
            selection_node = best_child
            
        return selection_node
        
    def expand(self, node):
        '''
        扩展阶段: 从未扩展的动作中随机选择一个创建新节点
        '''
        # 找出所有未扩展的候选位置
        unexpanded_coordinates = []
        for pos in node.next_locations:
            if pos not in node.childnodes:
                unexpanded_coordinates.append(pos)
        
        # 如果只剩一个未扩展位置，标记为完全扩展
        if len(unexpanded_coordinates) == 1:
            node.status = 1
            
        # 随机选择一个未扩展位置
        random_seed = random.randint(0, len(unexpanded_coordinates)-1)
        expand_node_coordinate = unexpanded_coordinates[random_seed]
        
        # 创建新节点
        expand_node = Node()
        
        # 设置新节点颜色
        if node.parent is None:  # 根节点，扩展的子节点颜色相同
            expand_node.color = self.color
        else:  # 切换颜色
            expand_node.color = 'O' if node.color == 'X' else 'X'
            
        # 设置节点属性
        expand_node.parent = node
        expand_node.coordinate = expand_node_coordinate
        
        # 更新父节点
        node.childnodes.append(expand_node_coordinate)
        node.child.append(expand_node)
        
        return expand_node
       
    def simulation(self, node):
        '''
        模拟阶段: 从当前节点快速随机模拟到游戏结束
        '''
        # 创建棋盘副本
        board_simulation = copy.deepcopy(node.board)
        switch = 0  # 双方连续不能下子的计数
        
        # 随机模拟直到游戏结束
        while switch < 2:
            available_moves = board_simulation.locations()
            if not available_moves:  # 无子可走
                switch += 1
            else:
                # 随机选择一步棋
                random_move = available_moves[random.randint(0, len(available_moves)-1)]
                board_simulation.reversi_pieces(random_move)
                switch = 0  # 重置计数器
                
            # 切换玩家
            board_simulation.color = 'O' if board_simulation.color == 'X' else 'X'
        
        # 计算胜负
        board_simulation.pieces_index()
        return board_simulation.win()
       
    def back_update(self, node):
        '''
        反向传播阶段: 更新节点及其祖先节点的统计信息
        '''
        current_score = node.score
        current_node = node
        
        while current_node.parent:
            current_node.parent.visit += 1
            # 分数在不同层级间反转，因为是零和游戏
            current_score = -current_score
            current_node.parent.score += current_score
            current_node = current_node.parent
            
    def mcts_run(self):
        '''
        执行完整的MCTS搜索过程
        
        返回:
            最佳行动坐标 (i,j)
        '''
        # 创建根节点
        root = Node()
        root.color = self.color
        root.board = self.board
        root.board.color = root.color
        root.next_locations = root.board.locations()
        
        # 检查是否有合法落子
        if not root.next_locations:
            return None
        
        # 第一次扩展
        expand_node = self.expand(root)
        
        # 创建并更新扩展节点的棋盘状态
        board_copy = copy.deepcopy(self.board)
        board_copy.reversi_pieces(expand_node.coordinate)
        expand_node.board = board_copy
        expand_node.visit = 1
        
        # 设置正确的棋盘颜色并模拟
        opponent_color = 'O' if expand_node.color == 'X' else 'X'
        expand_node.board.color = opponent_color
        
        # 根据颜色正确处理模拟结果
        if expand_node.color == 'X':
            expand_node.score = self.simulation(expand_node)
        else:
            expand_node.score = -self.simulation(expand_node)
            
        # 更新扩展节点的可行位置
        expand_node.next_locations = expand_node.board.locations()
        
        # 反向传播
        self.back_update(expand_node)
        
        # 主搜索循环
        iterations = 0
        while iterations < self.r and expand_node.next_locations:
            iterations += 1
            
            # 选择-扩展-模拟-反向传播
            selection_node = self.selection(root)
            expand_node = self.expand(selection_node)
            
            # 创建并更新扩展节点的棋盘状态
            board_copy = copy.deepcopy(selection_node.board)
            board_copy.reversi_pieces(expand_node.coordinate)
            expand_node.board = board_copy
            expand_node.visit = 1
            
            # 设置正确的棋盘颜色并模拟
            opponent_color = 'O' if expand_node.color == 'X' else 'X'
            expand_node.board.color = opponent_color
            
            # 根据颜色正确处理模拟结果
            if expand_node.color == 'X':
                expand_node.score = self.simulation(expand_node)
            else:
                expand_node.score = -self.simulation(expand_node)
                
            # 更新扩展节点的可行位置
            expand_node.next_locations = expand_node.board.locations()
            
            # 反向传播
            self.back_update(expand_node)
        
        # 选择访问次数最多的动作
        action = None
        max_visit = 0
        for child in root.child:
            if child.visit > max_visit:
                max_visit = child.visit
                action = child.coordinate
                
        return action