COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

# 定义8个方向向量：上、下、左、右、左上、右上、左下、右下
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.transposition_table = {}  # 添加转置表提高搜索效率

    def go(self, chessboard):
        """使用迭代加深搜索寻找最佳下一步"""
        # 先找出所有合法走法
        self.candidate_list = self.find_choice(chessboard, self.color)
        
        # 设置Alpha-Beta剪枝初始值
        alpha = -99999
        beta = 99999
        
        # 清空转置表，避免不同深度间干扰
        self.transposition_table = {}
        
        # 迭代加深搜索，保持与原代码相同的深度
        for depth in [2, 3, 4, 6]:
            move = self.search(self.color, chessboard, 1, alpha, beta, depth)
            if move is not None:
                self.candidate_list.append(move)

    def search(self, color, board, deep, alpha, beta, t):
        """Alpha-Beta搜索"""
        # 达到搜索深度限制时返回评估值
        if deep == t:
            choice = self.find_choice(board, self.color)
            un_choice = self.find_choice(board, -self.color)
            return self.assess(board) + (len(choice) - len(un_choice)) * 220
        
        # 检查转置表中是否已有此局面的结果
        board_hash = self._get_board_hash(board)
        if board_hash in self.transposition_table:
            stored_depth, stored_value, stored_move = self.transposition_table[board_hash]
            if stored_depth >= t - deep:  # 如果已存储的结果搜索深度足够
                if deep == 1:
                    return stored_move
                return stored_value
        
        # 获取所有合法走法
        choice = self.find_choice(board, color)
        if not choice:
            return None
            
        # MAX层处理（奇数层）
        if deep % 2 == 1:
            a = -99999  # 最佳值
            b = 0       # 最佳移动索引
            
            for i in range(len(choice)):
                # 模拟此移动
                newboard = self.find_change(board, color, choice[i])
                # 递归搜索下一层
                n = self.search(-color, newboard, deep+1, alpha, beta, t)
                
                if n is not None:
                    if n > a:
                        a = n
                        b = i
                    # Alpha剪枝
                    if a >= beta:
                        # 存储到转置表
                        self.transposition_table[board_hash] = (t-deep, a, choice[i] if deep == 1 else None)
                        if deep == 1:
                            return choice[i]
                        return a
                    alpha = max(alpha, a)
            
            # 修复：确保在有合法走法时才返回最佳走法
            if len(choice) > 0:
                # 记录结果到转置表
                self.transposition_table[board_hash] = (t-deep, a, choice[b] if deep == 1 else None)
                if deep == 1:
                    return choice[b]
                return a
            else:
                return None
            
        # MIN层处理（偶数层）
        else:
            a = 99999   # 最佳值
            b = 0       # 最佳移动索引
            
            for i in range(len(choice)):
                # 模拟此移动
                newboard = self.find_change(board, color, choice[i])
                # 递归搜索下一层
                n = self.search(-color, newboard, deep+1, alpha, beta, t)
                
                if n is not None:
                    if n < a:
                        a = n
                        b = i
                    # Beta剪枝
                    if a <= alpha:
                        # 存储到转置表
                        self.transposition_table[board_hash] = (t-deep, a, choice[i] if deep == 1 else None)
                        if deep == 1:
                            return choice[i]
                        return a
                    beta = min(beta, a)
            
            # 修复：确保在有合法走法时才返回最佳走法
            if len(choice) > 0:
                # 记录结果到转置表
                self.transposition_table[board_hash] = (t-deep, a, choice[b] if deep == 1 else None)
                if deep == 1:
                    return choice[b]
                return a
            else:
                return None

    def _get_board_hash(self, board):
        """将棋盘转换为哈希用于转置表"""
        return tuple(tuple(row) for row in board)

    def find_change(self, chessboard, color, choice):
        """使用方向向量优化的落子翻转函数"""
        # 创建棋盘副本
        change = [row[:] for row in chessboard]  # 深拷贝二维列表
        x, y = choice
        change[x][y] = color  # 在选定位置放置棋子
        
        # 检查8个方向
        for dx, dy in DIRECTIONS:
            # 确认第一个相邻位置是对手棋子
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.chessboard_size and 0 <= ny < self.chessboard_size):
                continue
            if chessboard[nx][ny] != -color:
                continue
            
            # 沿该方向查找自己的棋子
            x_temp, y_temp = nx + dx, ny + dy
            can_flip = False
            
            while 0 <= x_temp < self.chessboard_size and 0 <= y_temp < self.chessboard_size:
                if chessboard[x_temp][y_temp] == 0:  # 遇到空位
                    break
                if chessboard[x_temp][y_temp] == color:  # 找到自己的棋子
                    can_flip = True
                    break
                x_temp += dx
                y_temp += dy
            
            # 如果找到封闭序列，翻转中间所有对手棋子
            if can_flip:
                flip_x, flip_y = nx, ny
                while flip_x != x_temp or flip_y != y_temp:
                    change[flip_x][flip_y] = color
                    flip_x += dx
                    flip_y += dy
        
        return change

    def find_choice(self, chessboard, color):
        """使用方向向量优化的合法走法查找"""
        choices = []
        # 遍历棋盘
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                # 只考虑空位置
                if chessboard[i][j] != 0:
                    continue
                
                is_valid = False
                # 检查8个方向
                for dx, dy in DIRECTIONS:
                    # 检查第一个相邻位置是否是对手颜色
                    nx, ny = i + dx, j + dy
                    if not (0 <= nx < self.chessboard_size and 0 <= ny < self.chessboard_size):
                        continue
                    if chessboard[nx][ny] != -color:
                        continue
                    
                    # 沿该方向查找自己的棋子
                    x_temp, y_temp = nx + dx, ny + dy
                    while 0 <= x_temp < self.chessboard_size and 0 <= y_temp < self.chessboard_size:
                        if chessboard[x_temp][y_temp] == 0:  # 遇到空位
                            break
                        if chessboard[x_temp][y_temp] == color:  # 找到自己的棋子
                            choices.append((i, j))
                            is_valid = True
                            break
                        x_temp += dx
                        y_temp += dy
                    
                    if is_valid:
                        break  # 只要有一个方向符合条件就可以了
        
        return choices

    def assess(self, board):
        """评估函数，保持原有策略同时优化计算"""
        # 位置权重表
        assess = [
            [2000, -60,  300, 200, 200, 300, -60,  2000],
            [-60, -400,    1,   1,   1,   1, -400,  -60],
            [300,    1,   10,   5,   5,  10,    1,  300],
            [200,    1,    5,   3,   3,   5,    1,  200],
            [200,    1,    5,   3,   3,   5,    1,  200],
            [300,    1,   10,   5,   5,  10,    1,  300],
            [-60, -400,    1,   1,   1,   1, -400,  -60],
            [2000, -60,  300, 200, 200, 300, -60,  2000]
        ]
        
        # 计算已占用格子数量
        status = sum(1 for i in range(8) for j in range(8) if board[i][j] != 0)
        
        # 角落已占领时的特殊调整
        if board[0][0] != 0:
            if board[0][1] == board[0][0]:
                assess[0][1] = 400
            if board[1][0] == board[0][0]:
                assess[1][0] = 400
            if board[0][1] == board[0][0] and board[1][0] == board[0][0] and board[1][1] == board[0][0]:
                assess[1][1] = 100
                
        if board[0][7] != 0:
            if board[0][6] == board[0][7]:
                assess[0][6] = 400
            if board[1][7] == board[0][7]:
                assess[1][7] = 400
            if board[0][6] == board[0][7] and board[1][7] == board[0][7] and board[1][6] == board[0][7]:
                assess[1][6] = 100
                
        if board[7][0] != 0:
            if board[6][0] == board[7][0]:
                assess[6][0] = 400
            if board[7][1] == board[7][0]:
                assess[7][1] = 400
            if board[6][0] == board[7][0] and board[7][1] == board[7][0] and board[6][1] == board[7][0]:
                assess[6][1] = 100
                
        if board[7][7] != 0:
            if board[6][7] == board[7][7]:
                assess[6][7] = 400
            if board[7][6] == board[7][7]:
                assess[7][6] = 400
            if board[7][6] == board[7][7] and board[6][7] == board[7][7] and board[6][6] == board[7][7]:
                assess[6][6] = 100
                
        # 游戏后期策略调整
        if status > 60:
            assess = [[500 for _ in range(8)] for _ in range(8)]

        # 计算总评分
        result_sum = 0
        for i in range(8):
            for j in range(8):
                result_sum += assess[i][j] * board[i][j]
                
        # 根据AI颜色调整分数
        if self.color == -1:
            result_sum = -result_sum
            
        return result_sum