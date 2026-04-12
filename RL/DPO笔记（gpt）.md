
# DPO 笔记：从提出动机到算法原理

## 1. DPO 是为了解决什么问题提出的？

在传统 RLHF 流程里，通常是：

1. 收集人类偏好数据
2. 训练一个 **Reward Model**
3. 再用 PPO 等强化学习方法去优化策略

这套方法虽然有效，但有几个明显问题：

- **流程复杂**：需要 SFT 模型、Reference Model、Reward Model、Critic，通常是“四个模型”
- **训练不稳定**：RL 本身就比监督学习更难训
- **显存和工程成本高**
- **需要在线采样 / rollout** ==(online)模型==
- **PPO 等方法需要处理 advantage、value、KL penalty、sampling variance 等问题**

于是 DPO（Direct Preference Optimization）提出了一个关键问题：

> 能不能跳过“先学 reward，再做 RL”这两步，直接从偏好数据学策略？

---

## 2. 为什么不用直接给回答打绝对分数，而更喜欢偏好排序？

人类标注时，更常见的不是：

- “这个回答值 8.3 分”
- “那个回答值 6.7 分”

而是：

- “在这两个回答里，我更喜欢 A，不喜欢 B”

原因很自然：

### 2.1 绝对分数很主观
不同人对“8分”标准不同：

- 有人很宽松
- 有人很严格
- 同一个人不同时间标准也可能变

所以绝对分数的一致性较差。

### 2.2 相对偏好更稳定
虽然人们不一定能一致地给出“几分”，  
但通常更容易在两个回答之间判断：

> 哪个更好？

也就是说，人类更擅长提供 **pairwise preference**：

$$
(x, y_w, y_l)
$$

其中：

- $x$：prompt
- $y_w$：winner（更喜欢的回答）
- $y_l$：loser（不喜欢的回答）

所以 DPO 和 Reward Model 训练都建立在 **偏好对数据** 上，而不是绝对分数上。

---

## 3. 为什么要引入 Bradley-Terry？

因为偏好数据只告诉我们：

> $y_w$ 比 $y_l$ 好

但模型训练需要一个 **可微、可优化的概率目标**。  
于是引入 Bradley-Terry 模型，把“谁更好”写成一个概率。

---

### 3.1 Bradley-Terry 是什么？

它假设每个回答 $y$ 都有一个隐含分数：

$$
r(x,y)
$$

那么对于两个回答 $y_w$ 和 $y_l$，人类更偏好 $y_w$ 的概率写成：

$$
P(y_w \succ y_l \mid x)
=
\frac{e^{r(x,y_w)}}{e^{r(x,y_w)} + e^{r(x,y_l)}}
$$

$$
= P(y_w \succ y_l \mid x)
=
\frac{e^{r(x,y_w)-r(x, y_l)}}{e^{r(x,y_w)-r(x, y_l)} + 1}
$$
等价地：

$$
P(y_w \succ y_l \mid x)
=
\sigma(r(x,y_w)-r(x,y_l))
$$

其中$\sigma(\cdot)$是 sigmoid。

---

### 3.2 为什么这很重要？

因为这就把：

- “偏好数据”
- “reward difference”
- “训练 loss”

连起来了。

于是传统 Reward Model 的训练 loss 就是：

$$
L_{RM}
=
-\log \sigma(r(x,y_w)-r(x,y_l))
$$

也就是说，Bradley-Terry 本质上就是一个 **偏好概率模型**。

---

## 4. DPO 最妙的洞察：reward 和最优策略有闭式关系

这一步是 DPO 最核心、最漂亮的地方。

传统 RLHF 的思路是：

$$
\text{偏好数据} \rightarrow \text{Reward Model} \rightarrow \text{RL} \rightarrow \pi_\theta
$$

但 DPO 论文发现：

> 在带 KL 正则的 RLHF 目标下，reward 和最优策略 $\pi^*$ 之间有闭式对应关系。

---

### 4.1 从 KL-regularized RL 目标出发

DPO 所基于的 RLHF 目标是：

$$
\max_\pi \mathbb{E}_{y\sim \pi(\cdot|x)}[r(x,y)] - \beta\, D_{KL}(\pi(\cdot|x)\|\pi_{ref}(\cdot|x))
$$

这里：

- $r(x,y)$：reward
- $\pi_{ref}$：reference policy
- $\beta$：控制 KL 正则强度

这个目标表示：

> 一边追求高 reward，一边不要偏离 reference policy 太远。

---

### 4.2 最优策略的闭式形式

这个问题的最优解满足：
==成正比关系==
$$
\pi^*(y|x) \propto \pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

整理一下可得：

$$
r(x,y)
=
\beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

这里 $Z(x)$ 是归一化项。

---

### 4.3 这一步为什么这么妙？

因为它说明：

> **reward 不是一个必须单独显式学习的独立对象**
>
> 它可以直接由“最优策略相对 reference 的偏移”来表示。

也就是说：

- 以前：先学 reward，再学策略
- 现在：==直接学策略，reward 被隐式吸收进 policy 里==

这就是 DPO 为什么能“消掉 reward model”的根本原因。

---

## 5. 为什么 DPO 不需要 Reward Model？

不是因为 reward 不重要了，  
而是因为 **reward 被 policy ratio 隐式表示了**。

根据上面的式子：

$$
r(x,y)
=
\beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

DPO 做的事情是：

- 用参数化策略 $\pi_\theta$ 去近似未知的最优策略 $\pi^*$
- 所以把上式中的 $\pi^*$ 换成待训练的 $\pi_\theta$

得到隐式 reward：

$$
r_\theta(x,y)
=
\beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

于是 Reward Model 不再需要单独训练。

---

## 6. 为什么 DPO 不需要 Critic？

Critic 的作用主要是在 PPO / Actor-Critic 里估计：

- $V(s)$
- advantage
- value baseline

因为 PPO 是 RL，需要估计回报、优势函数、降低方差。

但 DPO 不再做 RL rollout optimization，  
而是直接优化一个从偏好对推导出来的监督式 loss。

所以：

- 不需要 value
- 不需要 advantage
- 不需要 critic

因此 DPO 最终只需要：

- **Actor / Policy**
- **Reference Policy**

---

## 7. DPO 的核心 loss 是怎么来的？

前面已经有两块拼图：

1. Bradley-Terry：
$$
P(y_w \succ y_l \mid x)
=
\sigma(r(x,y_w)-r(x,y_l))
$$

2. 隐式 reward：
$$
r_\theta(x,y)
=
\beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

把第二个代入第一个：

$$
P(y_w \succ y_l \mid x)
=
\sigma\left(
r_\theta(x,y_w)-r_\theta(x,y_l)
\right)
$$

展开：

$$
=
\sigma\left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
$$

因为同一个 $x$ 下，$\beta \log Z(x)$ 会相消。

于是 DPO loss 为：

$$
L_{DPO}
=
-\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
$$

这就是 DPO 的核心训练目标。


==Q: 为什么损失要取log?==
A: 因为如果直接优化$-\sigma$ ，当winner的reward比loser的reward大很多的时候，根据`sigmoid`的函数图像，可以推出梯度比较小，这是符合直觉的，因为预测的比较准确，参数不需要大改。但如果win的reward比loss小很多，也就是sigmoid里面是负的很多，那么此时梯度仍然是一个比较小的值，这显然是不对的，因为预测和我们想要的差距很大，此时应该大步更新，用-log后，sigmoid在0附近，-log后值为$-\infty$，这个时候梯度还是比较大的。

---

## 8. DPO 中的 KL 正则体现在哪里？

DPO 没有显式写：

$$
\lambda D_{KL}(\pi_\theta\|\pi_{ref})
$$

这样的惩罚项。

所以它不是“显式 KL penalty”，而是：

> **KL 信息已经被编码进了 policy ratio 里**

具体体现在：

$$
\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

这个量中。

---

### 8.1 为什么说这里包含 KL 正则信息？

因为最初的 RLHF 目标本来就是：

$$
\max_\pi \mathbb{E}[r] - \beta KL(\pi\|\pi_{ref})
$$

最优策略推导出来后：

$$
\pi^*(y|x)\propto \pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

说明新策略不是凭空优化的，而是：

- 以 $\pi_{ref}$ 为底座
- 按 reward 做指数重加权

所以 DPO 不是“无约束地把 winner 概率拉高”，  
而是：

> **相对于 reference，适度提升 winner、压低 loser**

因此 DPO 的 KL 是 **隐式正则**。

---

## 9. DPO 会不会一直无限拉大 $y_w$ 和 $y_l$ 的差距？

从 loss 形式上看：

$$
-\log \sigma(\Delta)
$$

其中：

$$
\Delta
=
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
$$

训练当然是在鼓励 $\Delta$ 变大。

所以可以说：

> DPO 确实有持续拉大 winner / loser margin 的倾向。

但注意几点：

### 9.1 不是绝对概率差，而是相对 ref 的 log-ratio 差
它不是单纯让 $\pi_\theta(y_w)$ 大、$\pi_\theta(y_l)$ 小，  
而是让它们相对于 $\pi_{ref}$ 的偏移产生区分。

### 9.2 sigmoid 有梯度饱和
$$
-\log \sigma(\Delta)
$$
当 $\Delta$ 已经很大时，梯度会变小。  
所以它不是线性地无限猛推，而是“越来越不在意”。

### 9.3 仍然可能过拟合
如果数据少、噪声大、训练过猛，DPO 仍然可能把某些偏好差异放大过头。

这也是后来 IPO 等方法提出的一个背景。

---

## 10. DPO 为什么可以看成监督学习问题？

因为它最后的训练形式已经不再是 RL，而是一个标准的 pairwise classification / ranking loss。

训练数据是：

$$
(x, y_w, y_l)
$$

标签含义是：

> 在这两个回答中，$y_w$ 比 $y_l$ 更好

优化目标是：

$$
-\log \sigma(\text{某个分差})
$$

这和很多监督学习里的二分类 / 排序学习非常像。

因此 DPO 可以看成：

> **一个用偏好对数据训练策略的监督学习问题**

它不需要：

- 在线 rollout
- policy gradient
- value estimation
- reward model inference + PPO

而是直接做：

- 前向计算 winner / loser 的 log-prob
- 计算 DPO loss
- 反向传播更新 policy

所以它在工程上非常稳定、简单、显存友好。

---

## 11. DPO 是 offline 还是 online？是 on-policy 还是 off-policy？

DPO 通常是：

- **offline**
- **off-policy**

---

### 11.1 为什么是 offline？

因为 DPO 训练时使用的是固定偏好数据集：

$$
(x, y_w, y_l)
$$

这些数据通常是提前收集好的，训练过程中不需要当前策略再去和环境实时交互，不需要在线生成新数据。

所以它是 **offline**。

---

### 11.2 为什么是 off-policy？

因为训练时使用的数据并不是当前最新策略 $\pi_\theta$ 现场采样得到的。

这些 winner / loser 对通常来自：

- SFT policy
- earlier model
- 人工采样
- 历史数据集

而不是当前正在更新的 $\pi_\theta$。

所以它不是 on-policy，而是 **off-policy**。

---

## 12. DPO 的算法流程

可以把 DPO 流程总结成下面几步：

### Step 1：准备一个 SFT / reference model
先有一个基础的参考策略 $\pi_{ref}$，一般就是 SFT 模型。

### Step 2：准备偏好数据
收集 pairwise preference 数据：

$$
(x, y_w, y_l)
$$

其中 $y_w$ 是 preferred response，$y_l$ 是 rejected response。

### Step 3：用当前策略计算 log-prob
分别计算：

$$
\log \pi_\theta(y_w|x), \quad \log \pi_\theta(y_l|x)
$$

同时 reference model 计算：

$$
\log \pi_{ref}(y_w|x), \quad \log \pi_{ref}(y_l|x)
$$

### Step 4：构造 DPO margin
$$
\Delta
=
\beta \left[
\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
$$

### Step 5：计算 loss
$$
L_{DPO} = -\log \sigma(\Delta)
$$

### Step 6：反向传播更新 $\pi_\theta$
只更新当前策略 $\pi_\theta$，reference model 固定不动。

### Step 7：训练结束后得到近似最优策略
最终得到的 $\pi_\theta$ 就是对理论最优策略 $\pi^*$ 的近似。

---

## 13. 一句话总结 DPO

DPO 的核心思想是：

> 利用 KL-regularized RL 中 reward 与最优策略的闭式关系，把原本“训练 reward model + 用 RL 训练 policy”的两阶段流程，压缩成一个基于偏好数据的监督式优化问题。

它的关键特点是：

- 用 **偏好对** 替代绝对分数
- 用 **Bradley-Terry** 把偏好写成概率
- 用 **policy ratio** 隐式表示 reward
- **不需要 reward model**
- **不需要 critic**
- **KL 正则信息被隐式编码进 $\log \frac{\pi_\theta}{\pi_{ref}}$ 中**
- 本质上是 **offline + off-policy**
- 最终训练形式很像 **监督学习中的二分类 / 排序学习**

---

## 14. 最后总结构图

$$
\text{人类更擅长相对偏好，而不是绝对打分}
$$

$$
\Downarrow
$$

$$
\text{收集 pairwise preference 数据 } (x, y_w, y_l)
$$

$$
\Downarrow
$$

$$
\text{用 Bradley-Terry 建模偏好概率}
$$

$$
P(y_w \succ y_l \mid x)=\sigma(r(x,y_w)-r(x,y_l))
$$

$$
\Downarrow
$$

$$
\text{DPO 论文证明 } r(x,y)
=
\beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta \log Z(x)
$$

$$
\Downarrow
$$

$$
\text{用 } \pi_\theta \text{ 近似 } \pi^*
$$

$$
\Downarrow
$$

$$
\text{reward model 被隐式吸收到 policy ratio 中}
$$

$$
\Downarrow
$$

$$
L_{DPO}
=
-\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
$$

$$
\Downarrow
$$

$$
\text{只训练 policy，得到近似最优策略}
$$

---

## 15. 适合速记的最终版本

**DPO 的提出背景**：  
传统 RLHF 需要 Reward Model + Critic + PPO，训练复杂且不稳定。与此同时，人类对“绝对打分”不一致，但对“哪个回答更好”的相对偏好更稳定，因此偏好学习通常采用 pairwise preference 数据。

**DPO 的关键做法**：  
先用 Bradley-Terry 把偏好写成概率模型，再利用 KL-regularized RL 中 reward 和最优策略的闭式关系，把 reward 改写成 policy 相对 reference 的 log-ratio，于是无需单独训练 Reward Model；同时由于不再需要 value / advantage 估计，也无需 Critic。

**DPO 的 KL 正则**：  
虽然损失函数中没有显式写出 KL penalty，但 KL 正则信息已经隐式编码在

$$
\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

这个 policy ratio 中。

**DPO 的训练形式**：  
使用固定偏好数据集，直接优化一个 pairwise logistic loss，因此它是 **offline、off-policy** 的，并且训练形式本质上很像一个监督学习中的二分类 / 排序学习问题。





## 我的怀疑
你的怀疑是对的：DPO 从带 KL 的原问题推导而来，所以 ref 的确被“隐式写进”了损失函数；但经过这些变形后，最终训练目标只显式优化 winner/loser 的相对 policy-ratio gap，并不显式约束整体 KL。因此它不能保证模型始终离 ref 不远。至于 β\betaβ，它并非无意义缩放，而是通过缩放 βg\beta gβg 改变 loss 的灵敏度和达到同样偏好置信度所需的 raw gap 大小，但它依然不是显式 KL 监控器