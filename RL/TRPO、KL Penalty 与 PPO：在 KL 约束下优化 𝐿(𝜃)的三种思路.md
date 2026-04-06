理论TRPO求解带约束的多元(\theta成亿上万亿)函数L()的一阶增量的极大值，使用拉格朗日求解

TRPO、KL penalty 和 PPO，本质上都在解决同一个问题：如何在“策略不要更新过猛”的前提下，尽可能增大 surrogate objective \(L(\theta)\)。

TRPO 的做法最直接：把 KL divergence 作为硬约束，要求新策略不能偏离旧策略超过某个 trust region。由于直接求解困难，它在旧参数点附近对 KL 做二阶近似，把约束写成
\(\frac12 \Delta\theta^T H \Delta\theta \le \delta\)，再在该安全区域内寻找使 \(L(\theta)\) 增大最快的更新方向。

KL penalty 则将这个约束“软化”为惩罚项，把目标改写为
\(L(\theta)-\beta KL\)。
此时 KL 不再是不能违反的约束，而变成“偏离越多、代价越大”的正则项。RLHF 中常见的 reference model KL penalty，本质上就属于这一类思路，只不过它被进一步写进 reward 里，作为 alignment regularizer。

PPO 则更进一步，为了实际部署的简洁性，不再把 KL 作为核心计算对象，而是通过 clipping policy ratio 的方式，隐式限制新旧策略变化幅度。也就是说，PPO 并不是直接优化 KL，而是用一阶的 clipping 机制近似 trust region 效果。

# TRPO、KL Penalty 与 PPO：KL 约束的三种处理方式，以及 PPO Clip Objective 的细致理解

---

## 1. 问题背景：为什么策略优化里总要管 KL

在策略优化中，我们希望更新后的策略比旧策略更好，因此想要尽可能增大 surrogate objective：

$$
L(\theta)
$$

但如果参数更新过猛，新策略可能会偏离旧策略太远，导致训练不稳定，甚至性能突然崩掉。

因此，核心问题变成了：

> **如何在“策略不要偏离太远”的前提下，尽可能增大 $L(\theta)$？**

这里，“偏离太远”通常就用 KL divergence 来衡量。

所以，TRPO、KL penalty、PPO 三者，本质上都在解决同一个问题：

$$
\text{在限制策略更新幅度的条件下，尽可能优化 } L(\theta)
$$

---

# 2. TRPO：把 KL 当成硬约束

TRPO 的建模最直接。它把问题写成一个约束优化：

$$
\max_\theta L(\theta)
\quad
\text{s.t. }
D_{KL}(\pi_{\theta_{\text{old}}}\|\pi_\theta)\le \delta
$$

含义很清楚：

- 目标：增大 $L(\theta)$
- 约束：新策略不能离旧策略超过一个 trust region
- 这个 trust region 由 KL divergence 定义

因此，TRPO 的思想可以概括为：

> **在 KL 定义的安全区域内，寻找让 $L(\theta)$ 上升最快的更新方向。**

---

## 2.1 为什么 TRPO 要对 KL 做二阶近似

直接求解上面的约束优化问题很难，因为 KL 是参数 $\theta$ 的复杂函数。  
于是 TRPO 在旧参数点附近，对 KL 做二阶近似。

令

$$
\theta = \theta_{\text{old}} + \Delta \theta
$$

那么在 $\theta_{\text{old}}$ 附近，

$$
D_{KL}(\pi_{\theta_{\text{old}}}\|\pi_{\theta_{\text{old}}+\Delta\theta})
\approx
\frac{1}{2}\Delta\theta^T H \Delta\theta
$$

其中 $H$ 可以理解为 KL 在旧点附近的 Hessian（更准确地说和 Fisher 信息矩阵有关）。

由于 KL 在 $\theta_{\text{old}}$ 处对自身的一阶导数为 0，所以局部没有一阶项，只剩下二次项。

与此同时，目标函数 $L(\theta)$ 在旧点附近可做一阶近似：

$$
L(\theta_{\text{old}}+\Delta\theta)
\approx
L(\theta_{\text{old}})+g^T\Delta\theta
$$

于是，TRPO 实际求解的是：

$$
\max_{\Delta\theta} g^T\Delta\theta
\quad
\text{s.t. }
\frac{1}{2}\Delta\theta^T H \Delta\theta \le \delta
$$

这相当于：

> **在一个二次型定义的安全椭球区域内，寻找使目标线性增益最大的更新方向。**

---

## 2.2 TRPO 的特点

TRPO 的 KL 处理方式是：

- **KL 是显式硬约束**
- **通过二阶近似，把 KL 约束写成二次型**
- **在安全范围内寻找最优上升方向**

优点：

- 理论严谨
- 直接体现 trust region 思想

缺点：

- 需要二阶信息
- 求解复杂，工程实现成本高

---

# 3. KL Penalty：把 KL 从“约束”变成“惩罚项”

TRPO 太复杂，于是出现了更容易实现的思路：

> 与其强行要求 KL 不超过阈值，不如允许偏离，但偏离越大，罚得越重。

于是问题改写为：

$$
\max_\theta \Big(L(\theta)-\beta D_{KL}(\pi_{\theta_{\text{old}}}\|\pi_\theta)\Big)
$$

这里：

- $L(\theta)$ 鼓励策略变好
- $-\beta KL$ 惩罚策略偏离过远
- $\beta$ 控制优化收益和稳定性之间的平衡

这时 KL 不再是“绝对不能违反的约束”，而变成：

> **你可以偏离，但你要为偏离付代价。**

---

## 3.1 KL Penalty 与 TRPO 的关系

TRPO 和 KL penalty 关注的是同一个核心矛盾：

$$
\text{既想增大 }L(\theta)\text{，又不想让策略变化太大}
$$

区别只在于表达方式：

### TRPO

$$
\max L(\theta)
\quad
\text{s.t. } KL \le \delta
$$

### KL Penalty

$$
\max \big(L(\theta)-\beta KL\big)
$$

也就是说：

- TRPO：**硬约束**
- KL penalty：**软惩罚**

从优化角度看，KL penalty 可以理解为对约束优化的一种拉格朗日松弛。

---

# 4. RLHF 中的 KL Penalty：把 KL 惩罚进一步写进 reward

在 RLHF 里，KL penalty 思想被进一步工程化。

我们常见的不是把 KL 单独写进优化目标，而是直接把它写进 reward 设计里：

$$
R = R_{\text{RM}} - \beta KL
$$

其中：

- $R_{\text{RM}}$：Reward Model 对整段回答的评分
- $-\beta KL$：对偏离 reference model 的惩罚

所以在 RLHF 中，KL 的角色更像是：

> **一种 reward shaping / alignment regularization**

也就是说：

- Reward Model 告诉模型：什么样的回答符合人类偏好
- Reference Model 的 KL 惩罚告诉模型：别偏离 SFT 初始分布太远

这里要特别注意：

> 这一步是在定义训练信号，不是在定义 PPO 算法本身。

---

# 5. PPO：不直接把 KL 当核心，而是用 clip ratio 近似 trust region

PPO 再进一步。它连显式 KL penalty 都不作为核心机制，而是采用更简单、更便宜的一阶方法：

> **直接限制新旧策略概率比值 $r_t(\theta)$ 的变化幅度。**

PPO 的核心目标函数是：

$$
L^{\text{clip}}(\theta)
=
\mathbb{E}\Big[
\min\big(
r_t(\theta)A_t,\;
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\big)
\Big]
$$

其中：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

---

## 5.1 PPO 的核心思想

PPO 不再显式解下面这个问题：

$$
\max L(\theta)
\quad
\text{s.t. } KL \le \delta
$$

而是换了一种方式：

- 不直接算 KL 约束
- 不做二阶近似
- 不解复杂约束优化
- 只控制概率比值 $r_t(\theta)$ 不要偏离太大

也就是说，PPO 的核心不是“直接优化 KL”，而是：

> **用 clipping 机制，隐式限制策略更新幅度，从而近似 trust region 的效果。**

---

# 6. 三者关系的本质总结

三者不是完全割裂的三种算法，而是同一个主线问题的三种处理思路：

---

## 6.1 TRPO

最理论、最严格：

$$
\max L(\theta)
\quad
\text{s.t. } KL \le \delta
$$

特点：

- 显式 KL 约束
- 二阶近似
- 理论漂亮
- 实现复杂

---

## 6.2 KL Penalty

对 TRPO 的松弛：

$$
\max \big(L(\theta)-\beta KL\big)
$$

特点：

- KL 从硬约束变软惩罚
- 更容易实现
- 仍然显式依赖 KL

---

## 6.3 PPO

更工程化：

$$
L^{\text{clip}}(\theta)
=
\mathbb E\Big[
\min(r_tA_t,\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
\Big]
$$

特点：

- 不直接把 KL 作为核心优化对象
- 用 ratio clipping 隐式限制更新幅度
- 一阶优化，简单稳定，易部署

---

## 6.4 一句话总结三者

> **TRPO 用 KL 作为硬约束，KL penalty 用 KL 作为软惩罚，PPO 则进一步绕开显式 KL，改用 clipping 机制来近似实现 trust region。**

它们的共同本质都是：

> **在限制策略不要偏离太远的前提下，尽可能增大 $L(\theta)$。**

---

# 7. PPO Clip Objective 的详细理解

PPO 的核心目标是：

$$
L^{\text{clip}}(\theta)
=
\mathbb E\Big[
\min\big(
r_t(\theta)A_t,\;
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\big)
\Big]
$$

其中：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

---

## 7.1 这个公式到底在做什么

初看这个公式时，最容易产生的疑问是：

> 如果左边的 $r_tA_t$ 更小，那么取 `min` 之后不还是选左边吗？  
> 这样真的有 clip 的作用吗？

关键点在于：

> **PPO 不是在所有情况下都强行截断，而是在“会导致目标过于乐观”的方向上做保守限制。**

也就是说，PPO 不是简单做：

$$
r_t \leftarrow \text{clip}(r_t,1-\epsilon,1+\epsilon)
$$

而是构造了一个 **更保守的 surrogate objective**。

---

## 7.2 Advantage 的符号决定 clip 截哪一边

先记住：

- 当 $A_t>0$ 时，这个动作是“好动作”，希望提高它的概率
- 当 $A_t<0$ 时，这个动作是“坏动作”，希望降低它的概率

因此：

- 对好动作，要防止“概率涨得过猛”
- 对坏动作，要防止“概率降得过猛”

PPO 的 clip 正是围绕这一点设计的。

---

# 8. 当 $A_t>0$ 时，clip 在干什么

当 $A_t>0$ 时，目标项为：

$$
\min\big(r_tA_t,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)
$$

由于 $A_t>0$，乘法不会改变大小关系，所以等价于：

$$
\min\big(r_t,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)\big)\cdot A_t
$$

---

## 8.1 如果 $r_t \in [1-\epsilon,1+\epsilon]$

此时 clip 不起作用：

$$
\text{clip}(r_t)=r_t
$$

两边相同，目标不变。

---

## 8.2 如果 $r_t > 1+\epsilon$

这表示：

> 对一个好动作，你把它的概率提高得太多了。

此时：

$$
\text{clip}(r_t)=1+\epsilon
$$

所以：

$$
\min(r_tA_t,(1+\epsilon)A_t)=(1+\epsilon)A_t
$$

也就是说：

> **对好动作，奖励它可以，但不能奖励过头。**

这就是 PPO 对正 advantage 情况下的截断作用。

---

## 8.3 如果 $r_t < 1-\epsilon$

这表示：

> 对一个好动作，你不但没有提升它的概率，反而把它的概率压低了。

此时：

$$
\text{clip}(r_t)=1-\epsilon
$$

而因为 $r_t < 1-\epsilon$，所以：

$$
r_tA_t < (1-\epsilon)A_t
$$

于是：

$$
\min(r_tA_t,(1-\epsilon)A_t)=r_tA_t
$$

也就是说，`min` 还是选左边。

这不是 bug，而是故意的。因为：

> PPO 只需要阻止“正确方向上冲得太猛”的更新，  
> 不需要去修复“本来就不好的更新”。

此时更新已经不乐观了，没必要额外截断。

---

# 9. 当 $A_t<0$ 时，clip 在干什么

当 $A_t<0$ 时，目标项仍是：

$$
\min\big(r_tA_t,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)
$$

但注意：

> 乘上负数后，大小关系会翻转。

---

## 9.1 如果 $r_t < 1-\epsilon$

这表示：

> 对一个坏动作，你把它的概率降得太狠了。

此时：

$$
\text{clip}(r_t)=1-\epsilon
$$

由于 $A_t<0$，可得：

$$
r_tA_t > (1-\epsilon)A_t
$$

所以：

$$
\min(r_tA_t,(1-\epsilon)A_t)=(1-\epsilon)A_t
$$

也就是说：

> **对坏动作，可以降低概率，但不能一下降得过猛。**

这就是 PPO 对负 advantage 情况下的截断作用。

---

## 9.2 如果 $r_t > 1+\epsilon$

这表示：

> 对一个坏动作，你反而提高了它的概率。

这是坏更新。此时 `min` 会直接保留那个更差的值，不会替你修正。

因为 PPO 的目标不是“帮你做对更新”，而是：

> **防止那些会让 surrogate objective 过于乐观的更新。**

---

# 10. PPO 的 clip 不是“统一截断”，而是“保守目标构造”

所以，PPO 的 `min` 并不是在做：

> “把所有 $r_t$ 都强行夹在区间里再优化”

它真正做的是：

> **在 unclipped objective 和 clipped objective 中，取一个更保守的值。**

这样构造出来的是一个 **pessimistic surrogate objective**。

---

# 11. 为什么 `min` 真的起到了 clip 作用

最容易困惑的点是：

> 如果左边已经更小，`min` 不还是选左边吗？clip 不是没起作用吗？

正确理解是：

> **clip 不要求在所有情况下都覆盖原目标，  
> 它只在“会导致过度乐观”的方向上生效。**

具体来说：

- 当 $A_t>0$ 时，只截断 $r_t>1+\epsilon$
- 当 $A_t<0$ 时，只截断 $r_t<1-\epsilon$

而：

- 当好动作被错误地下调时
- 当坏动作被错误地上调时

这些情况本身已经不乐观了，PPO 不需要再额外“帮你夹住”。

---

# 12. 最重要的一张表

| Advantage 符号 | 这个动作代表什么 | 希望概率怎么变 | 需要截断哪一边 |
|---|---|---|---|
| $A_t>0$ | 好动作 | 增大 | 截断 $r_t>1+\epsilon$ |
| $A_t<0$ | 坏动作 | 减小 | 截断 $r_t<1-\epsilon$ |

---

# 13. 一个数值例子

设：

$$
\epsilon=0.2
$$

---

## 例 1：$A_t=2>0$

### 情况 1：$r_t=1.5$

$$
r_tA_t = 1.5\times2 = 3
$$

$$
\text{clip}(r_t)=1.2
$$

$$
\text{clip}(r_t)A_t = 1.2\times2 = 2.4
$$

所以：

$$
\min(3,2.4)=2.4
$$

被截住了。

---

### 情况 2：$r_t=0.7$

$$
r_tA_t=0.7\times2=1.4
$$

$$
\text{clip}(r_t)=0.8
$$

$$
\text{clip}(r_t)A_t=0.8\times2=1.6
$$

所以：

$$
\min(1.4,1.6)=1.4
$$

没有被截。

原因是：这不是“奖励过头”，而是“奖励不够”，PPO 不需要管。

---

## 例 2：$A_t=-2<0$

### 情况：$r_t=0.6$

$$
r_tA_t=0.6\times(-2)=-1.2
$$

$$
\text{clip}(r_t)=0.8
$$

$$
\text{clip}(r_t)A_t=0.8\times(-2)=-1.6
$$

所以：

$$
\min(-1.2,-1.6)=-1.6
$$

被截住了。

这表示：对坏动作，概率下降可以，但不能降得太狠。

---

# 14. 最终总结

TRPO、KL penalty 和 PPO 本质上都在解决同一个问题：

> **如何在限制策略不要偏离太远的前提下，尽可能增大 $L(\theta)$。**

它们的区别在于对 KL 的处理方式不同：

- **TRPO**：把 KL 当作硬约束，并通过二阶近似在安全区域内找最优更新方向
- **KL penalty**：把 KL 从硬约束改成软惩罚，允许偏离但需要付代价
- **PPO**：进一步绕开显式 KL，不直接计算 KL，而是通过 clipping ratio 来隐式控制策略更新幅度

其中，PPO 的 clip objective 不是简单地“整体截断 $r_t$”，而是：

> **根据 $A_t$ 的正负，只在“会让目标过于乐观”的方向上做保守限制。**

因此：

- 对好动作，只防止概率涨得过猛
- 对坏动作，只防止概率降得过猛

这也是为什么 `min(unclipped, clipped)` 虽然不是每次都选 clipped，但仍然真正起到了 clip 的作用。