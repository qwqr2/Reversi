from player import AIPlayer, AIPlayerplus, ChessAIPlayer
from game import Game
from board import Board
import time
from policy_value_net import PolicyValueNet
from multiprocessing import Pool, Queue, Process
import numpy as np
import torch
import os

class ModelBattle:
    def __init__(self):
        try:
            # 检查模型文件是否存在
            if not os.path.exists('./current_policy.model'):
                print("警告：未找到模型文件 './current_policy.model'")
                print("将使用新初始化的模型")
                self.policy_value_net = PolicyValueNet(use_gpu=torch.cuda.is_available())
            else:
                # 尝试加载模型
                try:
                    self.policy_value_net = PolicyValueNet(model_file='./current_policy.model', use_gpu=torch.cuda.is_available())
                    print("成功加载模型")
                except Exception as e:
                    print(f"加载模型失败: {e}")
                    print("将使用新初始化的模型")
                    self.policy_value_net = PolicyValueNet(use_gpu=torch.cuda.is_available())
            
            self.models = {
                '1': ('神经网络+MCTS', lambda: AIPlayerplus(self.policy_value_net.policy_value_fn, 100)),
                '2': ('纯MCTS', lambda: AIPlayer(100)),
                '3': ('剪枝算法', lambda: ChessAIPlayer(4))
            }
        except Exception as e:
            print(f"初始化失败: {e}")
            raise
        
    def select_models(self):
        print("\n可用的AI模型：")
        for key, (name, _) in self.models.items():
            print(f"{key}. {name}")
        
        while True:
            try:
                black_choice = input("\n请选择黑棋AI模型 (输入数字): ")
                white_choice = input("请选择白棋AI模型 (输入数字): ")
                
                if black_choice in self.models and white_choice in self.models:
                    return (
                        self.models[black_choice][1](),
                        self.models[white_choice][1](),
                        black_choice,
                        white_choice
                    )
                else:
                    print("无效的选择，请重新输入")
            except Exception as e:
                print(f"选择出错: {e}")

    def run_battle(self, n_games=10):
        try:
            black_player, white_player, black_choice, white_choice = self.select_models()
            black_win = 0
            white_win = 0
            draw = 0
            black_times = []
            white_times = []
            
            print(f"\n开始{self.models[black_choice][0]} vs {self.models[white_choice][0]}的对战...")
            
            for i in range(n_games):
                print(f"\n第{i+1}局开始...")
                game = Game(black_player, white_player)
                
                # 记录决策时间
                start_time = time.time()
                game.selfplay_run()
                end_time = time.time()
                
                # 统计胜负
                result = game.board.win()
                if result == 1:
                    black_win += 1
                    print(f"黑棋({self.models[black_choice][0]})获胜")
                elif result == -1:
                    white_win += 1
                    print(f"白棋({self.models[white_choice][0]})获胜")
                else:
                    draw += 1
                    print("平局")
                
                # 记录决策时间
                game_time = end_time - start_time
                if i % 2 == 0:  # 黑棋先手
                    black_times.append(game_time)
                else:
                    white_times.append(game_time)
                
                # 显示当前进度
                print(f"当前进度: {i+1}/{n_games}")
                print(f"黑棋胜率: {black_win/(i+1)*100:.1f}%")
                print(f"白棋胜率: {white_win/(i+1)*100:.1f}%")
                print(f"平局率: {draw/(i+1)*100:.1f}%")
            
            # 输出最终统计结果
            print("\n对战统计结果：")
            print(f"总场次: {n_games}")
            print(f"黑棋({self.models[black_choice][0]})胜率: {black_win/n_games*100:.1f}%")
            print(f"白棋({self.models[white_choice][0]})胜率: {white_win/n_games*100:.1f}%")
            print(f"平局率: {draw/n_games*100:.1f}%")
            if black_times:
                print(f"黑棋平均决策时间: {np.mean(black_times):.2f}秒")
            if white_times:
                print(f"白棋平均决策时间: {np.mean(white_times):.2f}秒")
                
        except Exception as e:
            print(f"对战过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    try:
        battle = ModelBattle()
        while True:
            try:
                n_games = int(input("\n请输入对战局数: "))
                if n_games > 0:
                    break
                print("请输入大于0的数字")
            except ValueError:
                print("请输入有效的数字")
        
        battle.run_battle(n_games)
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()