from board import Board
import time

class Game(object):
    '''
    游戏规范
    '''
    def __init__(self, black_player, white_player):
        self.board = Board()      #棋盘
        self.black_player = black_player
        self.black_player.color = 'X'
        self.white_player = white_player
        self.white_player.color = 'O'
        self.current_player = self.black_player
        self.board.current_player = 'X'  # 确保board.current_player与当前玩家一致
        
        
    
    def switch_player(self):
        '''
        切换玩家
        '''
        if self.current_player == self.black_player:
            self.current_player = self.white_player
            self.board.color = 'O'
            self.board.current_player = 'O'  # 同步更新board.current_player
        else:
            self.current_player = self.black_player
            self.board.color = 'X'
            self.board.current_player = 'X'  # 同步更新board.current_player
        
    
    def run(self):
        '''
        游戏运行
        '''
        print('---------游戏现在开始！---------')
        self.board.display()
        
        
        print('-----黑棋先行-----')
        print('黑棋合法位置：', self.board.locations())
        print('请下黑棋（中间空格）:')
        start = time.time()

        self.board.pieces_index()  
        
        action = self.current_player.move(self.board)
        if action is not None:  # 确保有有效动作
            self.board.reversi_pieces(action)
        
        end = time.time()
        
        # self.board_list.append(self.board)          # 添加当前棋盘
        
        print('棋手思考时间：', int(end-start), '秒')
        print('--------------------------')
        self.board.display()
        print('--------------------------')
        self.switch_player()
        switch = 0
        while switch != 2:
            if len(self.board.locations()) == 0:
                print('无棋可走，对方下棋')
                switch += 1
            else:
                if self.current_player.color == 'X':
                    print('黑棋合法位置：', self.board.locations())
                    print('请下黑棋（中间空格）:')
                    switch = 0
                else:
                    print('白棋合法位置：', self.board.locations())
                    print('请下白棋（中间空格）:')
                    switch = 0
                  
                start = time.time()
                
                
                self.board.pieces_index()  
                
                action = self.current_player.move(self.board)
                if action is not None:  # 确保有有效动作
                    self.board.reversi_pieces(action)
                

                end = time.time()
                
                
                print('棋手思考时间：', int(end-start), '秒')
                print('--------------------------')
                self.board.display()  
                print('--------------------------')
            self.switch_player()
        print('--------游戏结束--------')
        self.board.pieces_index()
        self.board.show_pieces_index()
        
        if self.board.black_count > self.board.white_count:     #黑棋赢
            print('黑棋获胜！')
        elif self.board.black_count < self.board.white_count:       #白棋赢
            print('白棋获胜！')
        else:       #平局
            print('平局！')
        

    def selfplay_run(self):
        '''
        自我对抗时通过减少print来加速对抗运行
        用于检验胜率
        同时记录每一步的决策时间
        '''
        # 初始化记录每一步决策时间的列表
        black_move_times = []
        white_move_times = []
        
        # 第一步：黑棋
        self.board.locations()
        start_time = time.time()
        action = self.current_player.move(self.board)
        end_time = time.time()
        
        # 记录黑棋决策时间
        black_move_times.append(end_time - start_time)
        
        if action is not None:  # 确保有有效动作
            self.board.reversi_pieces(action)
        self.switch_player()
        
        # 后续步骤
        switch = 0
        while switch != 2:
            if len(self.board.locations()) == 0:
                switch += 1
            else:
                switch = 0
                self.board.locations()
                
                # 记录决策时间
                start_time = time.time()
                action = self.current_player.move(self.board)
                end_time = time.time()
                
                # 根据当前玩家记录决策时间
                if self.current_player == self.black_player:
                    black_move_times.append(end_time - start_time)
                else:
                    white_move_times.append(end_time - start_time)
                
                if action is not None:  # 确保有有效动作
                    self.board.reversi_pieces(action)
            self.switch_player()
        self.board.pieces_index()
        
        # 返回黑白棋的决策时间列表
        return black_move_times, white_move_times

    
    def selfplay_run_plus(self):
        '''
        自我对抗时通过减少print来加速对抗运行
        并且对棋盘数据进行收集
        '''
        self.board.locations()
        self.playdata_state = []
        self.playdata_prob = []
        self.playdata_who = []   # 当前玩家指示器，1为黑棋，0为白棋
        self.playdata_win = []
        action_and_mctsprob = self.current_player.move1(self.board)
        self.board.pieces_index()  
        self.playdata_state.append(self.board.current_state())
        self.playdata_prob.append(action_and_mctsprob[1])
        self.playdata_who.append(1)
        
        if action_and_mctsprob[0] is not None:  # 确保有有效动作
            self.board.reversi_pieces(action_and_mctsprob[0])
            
        self.board.pieces_index()                
        self.switch_player()
        switch = 0
        while switch != 2:
            if len(self.board.locations()) == 0:
                switch += 1
            else:
                switch = 0
                self.board.pieces_index() 
                self.board.locations()
                action_and_mctsprob = self.current_player.move1(self.board)               
                self.playdata_state.append(self.board.current_state())
                self.playdata_prob.append(action_and_mctsprob[1])
                if self.board.color == 'X':
                    self.playdata_who.append(1)
                else:
                    self.playdata_who.append(0)
                
                if action_and_mctsprob[0] is not None:  # 确保有有效动作
                    self.board.reversi_pieces(action_and_mctsprob[0])
                                
                self.board.pieces_index()
                self.board.locations()
                
            self.switch_player()
        self.board.pieces_index()
        
        win = self.board.win()
        
        for i in range(len(self.playdata_who)):
            if self.playdata_who[i] == 1:
                self.playdata_win.append(win) 
            else:
                self.playdata_win.append(-win)
        full_playdata = list(zip(self.playdata_state, self.playdata_prob, self.playdata_win))
        self.playdata = full_playdata

            
                    
            
        

