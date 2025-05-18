from player import HumanPlayer, AIPlayerplus, ChessAIPlayer
from game import Game
from policy_value_net import PolicyValueNet

policy_net_cache = {}

def get_policy_value_net(model_file_path):
    if model_file_path not in policy_net_cache:
        print(f"正在加载神经网络模型: {model_file_path} ...")
        policy_net_cache[model_file_path] = PolicyValueNet(model_file=model_file_path)
        print("模型加载完毕。")
    return policy_net_cache[model_file_path]

def select_player(player_name_prompt):
    """选择玩家类型"""
    print(f"\n请为 {player_name_prompt} 选择玩家类型：")
    print("1. 人类玩家")
    print("2. 神经网络 AI (AIPlayerplus)")
    print("3. 传统算法 AI (ChessAIPlayer)")
    
    choice = 0
    while choice not in [1, 2, 3]:
        try:
            choice_str = input(f"请选择玩家 (1-3): ")
            choice = int(choice_str)
            if choice not in [1, 2, 3]:
                print("无效的选择，请输入1, 2, 或 3。")
        except ValueError:
            print("请输入有效的数字 (1, 2, 或 3)!")
    
    if choice == 1:
        return HumanPlayer()
    elif choice == 2: # 神经网络 AI
        model_file = input("请输入神经网络模型文件路径 (默认: ./current_policy.model): ")
        if not model_file:
            model_file = './current_policy.model'
        try:
            policy_value_net = get_policy_value_net(model_file)
            mcts_playout_str = input("请输入MCTS搜索次数 (推荐: 400, 默认: 400): ")
            mcts_playout = int(mcts_playout_str) if mcts_playout_str.isdigit() else 400
            return AIPlayerplus(policy_value_net.policy_value_fn, mcts_playout)
        except Exception as e:
            print(f"创建神经网络AI失败: {e}")
            print("将使用人类玩家作为后备。")
            return HumanPlayer() # Fallback to HumanPlayer
    elif choice == 3: # 传统算法 AI
        search_depth_str = input("请输入传统算法搜索深度 (推荐: 4, 默认: 4): ")
        search_depth = int(search_depth_str) if search_depth_str.isdigit() else 4
        return ChessAIPlayer(search_depth)
    return None # Should not be reached

def main():
    print("欢迎来到黑白棋 AI 对战系统!")
    print("="*50)
    
    black_player = select_player("黑棋")
    if black_player is None: return # Exit if player creation failed

    white_player = select_player("白棋")
    if white_player is None: return # Exit if player creation failed
    
    print("\n游戏开始!")
    game = Game(black_player, white_player)
    game.run()
    
    # Game.run() already prints the winner based on its internal logic.
    # The winner determination here might be redundant if game.run() is comprehensive.
    # For clarity, let's assume game.run() handles end-game announcements.
    # If not, uncomment and adapt the following:
    # print("\n--- 游戏结束 ---")
    # winner = game.board.win() # Assuming board.win() returns 1 for black, -1 for white, 0 for tie
    # if winner == 1:
    #     print("黑棋 (X) 获胜!")
    # elif winner == -1:
    #     print("白棋 (O) 获胜!")
    # else:
    #     print("平局!")
        
    play_again = input("\n是否再玩一次? (y/n): ").lower()
    if play_again == 'y':
        main()
    else:
        print("感谢您的游玩！")

if __name__ == '__main__':
    main()