👉 RLHF 存在 reward model 不准确的问题，因此通常不能进行大规模或长期优化，否则会出现 reward hacking。

👉 RLVR 通过使用可验证的客观 reward（如正确/错误），在 certain tasks（如数学、代码）上提供更稳定的优化信号，减少 reward model 带来的偏差。

👉 但 RLVR 仍然面临 sparse reward、credit assignment 和 exploration 等问题，且仍可能出现 specification gaming，因此也不能无限制训练。

👉 当前最有效的范式是 SFT + RLHF + RLVR 的组合，而不是单一方法。

## RL在llm中的作用

一般pretrain和SFT都是对数据的模拟、拟合，原数据可以视为一个瓶颈，无论模型如何训练，都不会超过这个瓶颈。
但是RL不一样，RL通过与环境交互、得到奖励反馈给agent，agent根据反馈进行调整学习，如此循环。模型是有可能获得超越原数据的能力的，即AI战胜人类。

不同于玩 Atari 游戏或机器人控制，LLM 的环境（Prompt）是一次性的，且状态转移是确定性的（Deterministic Transitions）
$Z(x) = \sum_y \pi_{ref}(y|x) e^{\frac{1}{\beta} r(x,y)}$


在深度学习中，我们不直接对概率进行建模，而是对概率进行对数建模（化乘为加），这是可行的，因为对数函数是单调函数