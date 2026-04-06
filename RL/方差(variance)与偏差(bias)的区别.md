
## 状态泛化
多见于连续估计的情况，以前的离散情况都是网格模型，直接用列表列举所有$(s, a)$去计算Q、V值。而连续模型，显然不能穷尽所有s和a，因此引入函数去估计，对于状态动作对$(s, a)$输入到f中得到对应的值。所以对于训练中没见过的值，如果是离散情况就直接初始值0，显然非法。而连续情况的状态泛化比较好，即使之前没见过，也能得到一个相对准确的估计值。

## 方差

## 偏差（bias）
无偏估计:
$$E(\hat{\theta}) = E(\theta)$$
我们就说 $\hat{\theta}$是$\theta$的***无偏估计***

importance sampling 通过 ratio 精确修正分布差异，因此理论上无偏，但 ratio 的乘积（尤其在 llm长trajectory 级别中，每一个action都要乘一个ratio）会导致方差爆炸；TRPO 通过 KL 约束间接限制 ratio，使方差可控；PPO 则通过 clip 直接截断 ratio，引入偏差(即当ratio超过阈值直接夹断，类似$clamp(ratio, 1-\epsilon, 1+\epsilon）$）但显著降低方差，从而获得更稳定的训练。

🌟
在强化学习中，梯度的**高方差**主要来源于基于 Monte Carlo 的 trajectory 采样以及长序列下概率比ratio的乘积结构；
而**偏差**通常来源于算法设计中的近似（如 clipping、bootstrapping），这些近似通过引入一定偏差来显著降低方差，从而提升训练稳定性（典型如 PPO）。

### 🎯 Bias–Variance Tradeoff（RL版）

|方法|Bias|Variance|
|---|---|---|
|REINFORCE（MC）|低|高|
|TD / Q-learning|高|低|
|Actor-Critic|中等|中等|
|PPO（clip）|有偏|稳定|

# Q&A

## Q:为什么REINFORCE是低bias，高var

A: $\nabla J \sim R(\tau)\cdot \nabla \log \pi$

👉 关键点：$R(\tau) = r_0 + r_1 + \cdots$

👉 **是真实发生的回报（ground truth）**

## 🎯 本质

> ❗REINFORCE 不做任何近似  
> ❗不依赖模型  
> ❗不依赖估计值

👉 所以：$\mathbb{E}[R(\tau)] = \text{真实期望}$

👉 结果： ✅ **无偏（unbiased）**

---
 🧠 直觉理解就像
> “每次把整局打完，再总结经验”
👉 虽然 noisy，但❗**不会系统性错**


🖤
REINFORCE 使用 Monte Carlo return 作为梯度估计，其无偏性来源于 “sample return 的期望等于真实价值函数”；但由于每次 trajectory 都是随机采样且回报是长序列的累积，导致单个样本的波动较大，从而引入高方差。
TD 的偏差来自 bootstrap（用当前估计替代真实未来），而低方差来自只使用一步信息而非整条 trajectory。
🖤

## TD learning 为什么有bias？

A: 核心更新：  $Q(s,a) \leftarrow r + \gamma Q(s',a')$

---

👉 关键问题：

> ❗这里的 Q(s',a')是什么？

---

👉 答案：

> **你当前模型的估计**

---

## 🎯 本质问题

你在做：
用一个还没学准的 Q  
去更新另一个 Q

---

👉 这就是：

> ❗bootstrap（自举）

---

## 🚨 关键后果

如果：

Q(s',a') 偏高

👉 那：

Q(s,a) 也会被带高

---

👉 误差会传播：

误差 → 下一个状态 → 再传播

👉 这就是：

> ❗系统性偏差（bias）

## Q:为什么TD learning的方差比REINFORCE低？
A:因为：$r + \gamma Q(s')$

👉 只用一步 reward、一个估计去更新，而REINFORCE用的是整条trajectory所有步的累积reward，是累乘起来的，所以方差会很大

---

👉 不用整条 trajectory，所以：❗波动小 → 低方差

REINFORCE 使用 Monte Carlo 回报进行无偏估计但具有高方差；TD 方法通过 bootstrap 用当前价值函数估计未来回报，从而降低方差但引入偏差；baseline（尤其是状态价值函数 V(s)）通过去除公共回报成分降低方差而不改变期望；Advantage 将这一思想系统化，衡量动作的相对优劣；Actor-Critic 通过 TD 学习估计 Advantage；PPO 在此基础上通过 importance sampling 与 clipping 控制策略更新幅度，实现稳定高效的优化。