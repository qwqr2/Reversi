from mcts_plus import Mcts_plus
import random
import numpy as np
from chess import AI as chessAI # For ChessAIPlayer
from mcts import Mcts  # For pure MCTS AIPlayer

class HumanPlayer():
    '''
    人类玩家
    '''
    def move(self, board):
        '''
        玩家下棋
        '''
        action = None
        while True:
            try:
                # Assuming board.current_player holds 'X' or 'O'
                player_symbol = board.current_player if hasattr(board, 'current_player') else ""
                move_str = input(f"玩家 {player_symbol}, 请输入你的落子位置 (例如: 3 4): ")
                action_coords = tuple(map(int, move_str.split()))
                if len(action_coords) == 2 and action_coords in board.locations():
                    action = action_coords
                    break
                else:
                    print('输入不合法或位置无效，请重新输入。有效位置例如: (row col)')
            except ValueError:
                print('输入格式错误，请输入两个数字，以空格分隔。')
            except Exception as e:
                print(f"发生错误: {e}")
        return action

class AIPlayerplus():    # 利用结合神经网络的蒙特卡洛树搜索的AI玩家
    '''
    神经网络 AI 玩家
    '''
    def __init__(self, policy_value_function, mcts_n=400):
        self.mcts_n = mcts_n
        self.policy_value_function = policy_value_function
        
    def move(self, board):
        '''
        实际用 不传输mcts中数据
        '''
        if hasattr(board, 'pieces_index') and callable(getattr(board, 'pieces_index')):
            board.pieces_index()
        
        mcts_result = Mcts_plus(board, self.policy_value_function, self.mcts_n).mcts_run()
        
        if isinstance(mcts_result, tuple) and len(mcts_result) > 0:
            action = mcts_result[0]
        else: 
            action = mcts_result 

        valid_locations = board.locations()
        if not valid_locations: # No moves possible, should be handled by game logic (pass)
             # This player should ideally not be called if no moves are possible.
             # Returning None might be problematic if not handled by Game.run()
            return board.pass_action() if hasattr(board, 'pass_action') else None


        if action not in valid_locations:
            current_player_symbol = board.current_player if hasattr(board, 'current_player') else ""
            print(f"神经网络AI (玩家 {current_player_symbol}) MCTS未返回有效动作 ({action})，尝试随机选择。")
            action = random.choice(valid_locations) # Fallback
            
        return action
    
    def move1(self, board): # For self-play data collection if needed by training
        '''
        自我对战用 需要传输数据
        '''
        if hasattr(board, 'pieces_index') and callable(getattr(board, 'pieces_index')):
            board.pieces_index()

        # Assuming Mcts_plus with is_selfplay=1 or similar argument
        action_data = Mcts_plus(board, self.policy_value_function, self.mcts_n, is_selfplay=1).mcts_run() 
            
        return action_data

class ChessAIPlayer(): # Traditional Algorithm AI
    '''
    传统算法 AI 玩家 (使用 chess.py)
    '''
    def __init__(self, search_depth=4):
        self.search_depth = search_depth
        self.ai_engine = None
        self.board_size = 8 # Assuming a standard 8x8 board; adjust if necessary
        
    def move(self, board):
        '''
        根据当前局面返回下一步动作
        '''
        chess_board_representation = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        for r in range(self.board_size):
            for c in range(self.board_size):
                # Ensure board.board exists and is accessible
                if hasattr(board, 'board') and board.board[r][c] == 'X': 
                    chess_board_representation[r][c] = -1 
                elif hasattr(board, 'board') and board.board[r][c] == 'O': 
                    chess_board_representation[r][c] = 1  
        
        # Ensure board.current_player exists
        current_player_symbol = board.current_player if hasattr(board, 'current_player') else 'X' # Default if not found
        player_color_for_engine = -1 if current_player_symbol == 'X' else 1
        
        if self.ai_engine is None:
            self.ai_engine = chessAI(self.board_size, player_color_for_engine, 5) 
        else:
            if hasattr(self.ai_engine, 'color'):
                 self.ai_engine.color = player_color_for_engine
        
        if hasattr(self.ai_engine, 'candidate_list'):
            self.ai_engine.candidate_list = [] 
        
        alpha, beta = -float('inf'), float('inf')
        
        action = self.ai_engine.search(player_color_for_engine, chess_board_representation, 1, alpha, beta, self.search_depth)
        
        valid_locations = board.locations()
        if not valid_locations: # No moves possible
            return board.pass_action() if hasattr(board, 'pass_action') else None

        if action is None or not isinstance(action, tuple) or len(action) != 2 or action not in valid_locations:
            print(f"传统算法AI (玩家 {current_player_symbol}) 未找到有效动作或返回格式不正确 ({action})，使用随机策略。")
            action = random.choice(valid_locations) # Fallback
            
        return action
class AIPlayer():
    '''
    纯MCTS AI玩家（用于训练评估）
    '''
    def __init__(self, mcts_n=100):
        self.mcts_n = mcts_n
        
    def move(self, board):
        '''
        使用纯MCTS策略
        '''
        if hasattr(board, 'pieces_index') and callable(getattr(board, 'pieces_index')):
            board.pieces_index()
        
        # 使用纯MCTS获取行动
        action = Mcts(board, self.mcts_n).mcts_run()
        
        valid_locations = board.locations()
        if not valid_locations:
            return None
        
        if action not in valid_locations:
            current_player_symbol = board.current_player if hasattr(board, 'current_player') else ""
            print(f"纯MCTS AI (玩家 {current_player_symbol}) 未返回有效动作 ({action})，使用随机策略。")
            action = random.choice(valid_locations)
            
        return action
    
    