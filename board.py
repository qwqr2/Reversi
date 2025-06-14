import numpy as np

class Board(object):
    def __init__(self):
        '''
        棋盘初始化
        '''
        self.board = [['.' for _ in range(8)] for _ in range(8)]
        self.board[3][4] = 'X' # X为黑棋
        self.board[4][3] = 'X' # X为黑棋
        self.board[3][3] = 'O' # O为白棋
        self.board[4][4] = 'O' # O为白棋
        self.color = 'X'
        self.availables = []  # 添加availables属性
        self.current_player = 'X'  # 添加current_player属性，与color保持一致
        
    def display(self):
        '''
        打印棋盘
        '''
        for i in range(8):
            for j in range(8):
                print(self.board[i][j], end=' ')
            print('')
    
    def pieces_index(self):
        '''
        找寻黑白棋子位置并计数
        '''
        self.black_count = 0
        self.white_count = 0
        self.board1 = np.zeros((8,8))       #为当前玩家创建空的8x8数组
        self.board2 = np.zeros((8,8))       #为对方玩家创建空的8x8数组
        for i in range(8):
            for j in range(8):
                if self.color == 'X':
                    if self.board[i][j] == 'X':
                        self.board1[i][j] = 1
                        self.black_count = self.black_count + 1
                    elif self.board[i][j] == 'O':
                        self.board2[i][j] = 1
                        self.white_count = self.white_count + 1
                else:
                    if self.board[i][j] == 'O':
                        self.board1[i][j] = 1
                        self.white_count = self.white_count + 1
                    elif self.board[i][j] == 'X':
                        self.board2[i][j] = 1
                        self.black_count = self.black_count + 1

        
    def show_pieces_index(self):
        '''
        展示各棋数目
        '''
        print('黑棋总数：', self.black_count)
        print('白棋总数：', self.white_count)
        
    def pass_action(self):
        '''
        当没有合法走法时，返回一个表示"过"的特殊动作
        '''
        print(f"玩家 {self.current_player} 没有合法走法，跳过回合。")
        return None
        
    def locations(self):
        '''
        获取下棋合法位置并标记应反转棋子
        合法位置为详细信息
        合法位置1为简单坐标信息
        '''
        self.legal_loc = []
        self.legal_loc1 = []
        neighbor = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        if self.color == 'X':      #黑棋玩家
            for i in range(8):
                for j in range(8):
                    for dx, dy in neighbor:
                        x = i + dx
                        y = j + dy
                        if x >= 0 and x <= 7 and y >=0 and y <= 7 and self.board[i][j] == '.' and self.board[x][y] =='O':
                            if dx > 0 and dy > 0:
                                for k in range(1, min(8-x, 8-y)):
                                    if self.board[x+k][y+k] == 'X':
                                        self.legal_loc.append(((i,j),'bottomright',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x+k][y+k] =='O':
                                        pass
                                    else:
                                        break
                            elif dx > 0 and dy == 0:
                                for k in range(1, 8-x):
                                    if self.board[x+k][y] == 'X':        
                                        self.legal_loc.append(((i,j),'bottom',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x+k][y] =='O':
                                        pass
                                    else:
                                        break
                            elif dx > 0 and dy < 0:
                                for k in range(1, min(8-x, y+1)):
                                    if self.board[x+k][y-k] == 'X':
                                        self.legal_loc.append(((i,j),'bottomleft',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x+k][y-k] =='O':
                                        pass
                                    else:
                                        break
                            elif dx < 0 and dy > 0:
                                for k in range(1, min(x+1, 8-y)):
                                    if self.board[x-k][y+k] == 'X':
                                        self.legal_loc.append(((i,j),'topright',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x-k][y+k] =='O':
                                        pass
                                    else:
                                        break
                            elif dx < 0 and dy == 0:
                                for k in range(1, x+1):
                                    if self.board[x-k][y] == 'X':
                                        self.legal_loc.append(((i,j),'top',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x-k][y] =='O':
                                        pass
                                    else:
                                        break
                            elif dx < 0 and dy < 0:
                                for k in range(1, min(x+1, y+1)):
                                    if self.board[x-k][y-k] == 'X':
                                        self.legal_loc.append(((i,j),'topleft',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x-k][y-k] =='O':
                                        pass
                                    else:
                                        break
                            elif dx == 0 and dy > 0:
                                for k in range(1, 8-y):
                                    if self.board[x][y+k] == 'X':
                                        self.legal_loc.append(((i,j),'right',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x][y+k] =='O':
                                        pass
                                    else:
                                        break
                            elif dx == 0 and dy < 0:
                                for k in range(1, y+1):
                                    if self.board[x][y-k] == 'X':
                                        self.legal_loc.append(((i,j),'left',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x][y-k] =='O':
                                        pass
                                    else:
                                        break
                        else:
                            continue
            self.availables = list({}.fromkeys(self.legal_loc1).keys())  # 更新availables
            return self.availables
        else:
            for i in range(8):
                for j in range(8):
                    for dx, dy in neighbor:
                        x = i + dx
                        y = j + dy
                        if x >= 0 and x <= 7 and y >=0 and y <= 7 and self.board[i][j] == '.' and self.board[x][y] =='X':
                            if dx > 0 and dy > 0:
                                for k in range(1, min(8-x, 8-y)):
                                    if self.board[x+k][y+k] == 'O':
                                        self.legal_loc.append(((i,j),'bottomright',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x+k][y+k] =='X':
                                        pass
                                    else:
                                        break
                            elif dx > 0 and dy == 0:
                                for k in range(1, 8-x):
                                    if self.board[x+k][y] == 'O':        
                                        self.legal_loc.append(((i,j),'bottom',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x+k][y] =='X':
                                        pass
                                    else:
                                        break
                            elif dx > 0 and dy < 0:
                                for k in range(1, min(8-x, y+1)):
                                    if self.board[x+k][y-k] == 'O':
                                        self.legal_loc.append(((i,j),'bottomleft',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x+k][y-k] =='X':
                                        pass
                                    else:
                                        break
                            elif dx < 0 and dy > 0:
                                for k in range(1, min(x+1, 8-y)):
                                    if self.board[x-k][y+k] == 'O':
                                        self.legal_loc.append(((i,j),'topright',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x-k][y+k] =='X':
                                        pass
                                    else:
                                        break
                            elif dx < 0 and dy == 0:
                                for k in range(1, x+1):
                                    if self.board[x-k][y] == 'O':
                                        self.legal_loc.append(((i,j),'top',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x-k][y] =='X':
                                        pass
                                    else:
                                        break
                            elif dx < 0 and dy < 0:
                                for k in range(1, min(x+1, y+1)):
                                    if self.board[x-k][y-k] == 'O':
                                        self.legal_loc.append(((i,j),'topleft',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x-k][y-k] =='X':
                                        pass
                                    else:
                                        break
                            elif dx == 0 and dy > 0:
                                for k in range(1, 8-y):
                                    if self.board[x][y+k] == 'O':
                                        self.legal_loc.append(((i,j),'right',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x][y+k] =='X':
                                        pass
                                    else:
                                        break
                            elif dx == 0 and dy < 0:
                                for k in range(1, y+1):
                                    if self.board[x][y-k] == 'O':
                                        self.legal_loc.append(((i,j),'left',k))
                                        self.legal_loc1.append((i,j))
                                        break
                                    elif self.board[x][y-k] =='X':
                                        pass
                                    else:
                                        break
                        else:
                            continue
            self.availables = list({}.fromkeys(self.legal_loc1).keys())  # 更新availables
            return self.availables
        
    
    def reversi_pieces(self, action):
        '''
        反转棋子
        '''
        x = action[0]
        y = action[1]
        if self.color == 'X':
            self.board[x][y] = 'X'
            for i in self.legal_loc:
                if (x, y) == i[0]:
                    if i[1] == 'right':
                        for j in range(i[2]+1):
                            self.board[x][y+j] = 'X'
                    elif i[1] == 'left':
                        for j in range(i[2]+1):
                            self.board[x][y-j] = 'X'
                    elif i[1] == 'top':
                        for j in range(i[2]+1):
                            self.board[x-j][y] = 'X'
                    elif i[1] == 'bottom':
                        for j in range(i[2]+1):
                            self.board[x+j][y] = 'X'
                    elif i[1] == 'topleft':
                        for j in range(i[2]+1):
                            self.board[x-j][y-j] = 'X'
                    elif i[1] == 'bottomleft':
                        for j in range(i[2]+1):
                            self.board[x+j][y-j] = 'X'
                    elif i[1] == 'topright':
                        for j in range(i[2]+1):
                            self.board[x-j][y+j] = 'X'
                    elif i[1] == 'bottomright':
                        for j in range(i[2]+1):
                            self.board[x+j][y+j] = 'X'
        if self.color == 'O':
            self.board[x][y] = 'O'
            for i in self.legal_loc:
                if (x, y) == i[0]:
                    if i[1] == 'right':
                        for j in range(i[2]+1):
                            self.board[x][y+j] = 'O'
                    elif i[1] == 'left':
                        for j in range(i[2]+1):
                            self.board[x][y-j] = 'O'
                    elif i[1] == 'top':
                        for j in range(i[2]+1):
                            self.board[x-j][y] = 'O'
                    elif i[1] == 'bottom':
                        for j in range(i[2]+1):
                            self.board[x+j][y] = 'O'
                    elif i[1] == 'topleft':
                        for j in range(i[2]+1):
                            self.board[x-j][y-j] = 'O'
                    elif i[1] == 'bottomleft':
                        for j in range(i[2]+1):
                            self.board[x+j][y-j] = 'O'
                    elif i[1] == 'topright':
                        for j in range(i[2]+1):
                            self.board[x-j][y+j] = 'O'
                    elif i[1] == 'bottomright':
                        for j in range(i[2]+1):
                            self.board[x+j][y+j] = 'O' 
                                   
    def is_game_over(self):
        '''
        判断游戏是否结束
        当双方都没有合法落子位置时游戏结束
        '''
        # 保存当前颜色
        current_color = self.color
        
        # 检查黑棋是否有合法落子位置
        self.color = 'X'
        black_moves = self.locations()
        
        # 检查白棋是否有合法落子位置
        self.color = 'O'
        white_moves = self.locations()
        
        # 恢复原来的颜色
        self.color = current_color
        
        # 如果双方都没有合法落子位置，游戏结束
        return len(black_moves) == 0 and len(white_moves) == 0

    def win(self):
        '''
        判断胜负
        返回1表示黑棋胜，-1表示白棋胜，0表示平局
        '''
        self.pieces_index()
        if self.black_count > self.white_count:
            return 1
        elif self.black_count < self.white_count:
            return -1
        else:
            return 0
        
    def current_state(self):
        '''
        棋盘当前状态（包含当前选手棋盘和对方选手两个界面）
        '''
        current_state = np.dstack((self.board1,self.board2))
        current_state = current_state.transpose((2,0,1))
        return current_state
    

        
        