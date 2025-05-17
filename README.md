# Reversi AI - 黑白棋人工智能

本项目实现了一个结合了传统搜索算法与深度强化学习的黑白棋（Reversi/Othello）AI。玩家可以选择与不同类型的 AI 对战，包括基于 Alpha-Beta 剪枝的传统 AI 和基于策略价值网络的深度学习 AI。

## 特点

*   **多种 AI 对手**：
    *   **人类玩家**：手动操作。
    *   **传统算法 AI (`ChessAIPlayer`)**：基于 Alpha-Beta 剪枝搜索和静态棋局评估函数。
    *   **神经网络 AI (`AIPlayerplus`)**：采用策略与价值网络 (Policy-Value Network)，结合蒙特卡洛树搜索 (MCTS) 进行决策，通过自我对弈进行强化学习训练。
*   **模型训练**：提供 `train.py` 脚本，通过自我对弈 (self-play) 收集数据并训练神经网络模型。
*   **灵活对战**：
    *   `main.py`：在命令行进行人机或 AI 间的单局对战。
    *   `selfplay.py`：进行 AI 间的批量自动对战，方便评估模型性能和胜率。
*   **清晰的项目结构**：代码模块化，易于理解和扩展。

## 环境配置

*   Python (推荐 3.7+)
*   PyTorch (用于神经网络模型)
*   NumPy

你可以使用 pip 安装必要的库：
```bash
pip install torch numpy
```

## 文件结构说明

```
Reversi-main/
├── board.py            # 定义棋盘状态、规则和基本操作
├── game.py             # 定义游戏流程控制和玩家交互逻辑
├── player.py           # 定义不同类型的玩家 (人类, 传统AI, 神经网络AI)
├── mcts_plus.py        # 实现结合神经网络的蒙特卡洛树搜索 (MCTS)
├── policy_value_net.py # 定义神经网络模型 (策略价值网络)
├── train.py            # 神经网络模型的训练脚本
├── selfplay.py         # AI 自动对战脚本，用于评估和测试
├── main.py             # 命令行交互式对战主程序
├── chess.py           # 传统 AI 的核心算法 (评估函数, Alpha-Beta搜索)
├── current_policy.model # 当前训练的神经网络模型文件 (示例)
├── best_policy.model    # 训练过程中表现最佳的神经网络模型文件 (示例)
└── README.md           # 项目说明文件
```

## 使用说明

### 1. 训练神经网络模型 (可选)

如果你想从头开始训练或继续训练神经网络模型：

```bash
python train.py
```

*   训练脚本会通过自我对弈生成数据，并使用这些数据更新神经网络。
*   训练过程中会定期评估模型性能，并保存当前模型 (`current_policy.model`) 和迄今为止表现最佳的模型 (`best_policy.model`)。
*   你可以通过修改 `train.py` 中的参数 (如学习率、MCTS 模拟次数、训练轮数等) 来调整训练过程。

### 2. 进行命令行对战

运行 `main.py` 与 AI 或让 AI 之间进行对战：

```bash
python main.py
```

程序会提示你为黑棋和白棋选择玩家类型：
1.  人类玩家
2.  神经网络 AI (`AIPlayerplus`)
3.  传统算法 AI (`ChessAIPlayer`)

根据提示输入选择，并配置相应 AI 的参数 (如模型路径、MCTS 模拟次数、搜索深度等)。


