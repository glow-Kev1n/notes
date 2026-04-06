 
## 1、RL从对齐转向增强推理能力
进入 2024-2025 年，随着 **DeepSeek-V3/R1** 和 **Qwen2.5-Math** 等模型的发布，强化学习的焦点从“对齐人类价值观”转向了“增强复杂推理能力”。这一转变催生了基于**群体相对（Group Relative）**的优化算法，它们在数学和代码任务上展现了惊人的效果。

## 2、如何理解熵塌缩？

## 3、如何理解GRPO与PPO中的Token 级的裁剪（Token-level Clipping）以及GSPO更进一步，将优化粒度转移到序列级别，从根本上降低了高方差和结构性噪声
是不是：
$$\frac{\pi_\theta（a|s）}{\pi_{old}(a|s)}$$
这种针对于具体prefix和具体的token生成action，每次clip都是对单个策略动作进行clip，即对于同一策略的不同状态动作对clip的可能不一样？


你可以把它理解成：

> PPO / GRPO 在 token 级别上问：  
> “新策略相对旧策略，对这个具体 prefix 下这个具体 token，到底提高/降低了多少概率？这个增减是不是太猛了？”

## 4、为什么不直接 token-level advantage？

因为现实中： 很多任务的 reward 根本不是 token-level 的

比如：
- 数学题：只知道最后对不对
- 代码：只知道是否通过测试
- RLHF：人类只评整段回答

👉 你根本不知道：

> 第 17 个 token贡献了多少 reward

所以只能：

> **把整个 reward 平均“甩锅”给所有 token**


## 5、如何理解硬裁剪（Hard Clipping）是一个非连续的操作：一旦越界，梯度突然截断为 0。
难道PPO、GRPO里面clip一旦越界梯度为0，就不更新了？不是说只是限制更新步幅么？会拉一把但不至于不更新吧

$L_{SAPO} = - g_t \cdot A_t$。