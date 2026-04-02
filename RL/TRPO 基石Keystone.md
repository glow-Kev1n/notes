
J 是真实目标但难优化 → 用 L 近似  
L 只在当前点方向正确 → 需要限制步长  
用 KL 控制策略变化 → 保证近似有效  
最终得到：在安全范围内最大化局部收益提升
# surrogate objective
用$L(\theta)$ 去近似我们真正要优化的$J(\theta)$

## 🎯 在 θ = θ_old 附近：
$$
\nabla_\theta L(\theta) = \nabla_\theta J(\theta)
$$

# TRPO 中 J(θ) 与 L(θ) 的关系 + 核心理解总结

---

# 🧠 1. 核心结论（先记住）

> $$L(\theta) \neq J(\theta)$$  
> 但仅在当前点附近：  
> $$\nabla_\theta L(\theta) = \nabla_\theta J(\theta) \quad \text{at } \theta = \theta_{old}$$

👉 **L 是 J 的“局部可优化替代品（surrogate objective）”**

---

# 🔥 2. 原始目标函数：J(θ)

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

特点：

- 真实目标（我们最终想优化的）
- 分布依赖 θ（难算）
- 无法直接用旧数据优化

---

# 🚀 3. surrogate objective：L(θ)

TRPO 构造：

$$
L(\theta) =
\mathbb{E}_{\pi_{old}}
\left[
\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A_t
\right]
$$

👉 也就是：

- ratio × advantage
- 使用 importance sampling
- 用旧数据估计新策略表现

---

# 🎯 4. J 和 L 的本质关系

---

## ✅ 关键性质：

$$
\nabla_\theta L(\theta) = \nabla_\theta J(\theta)
\quad \text{at } \theta = \theta_{old}
$$

---

## 🧠 含义：

- 在当前策略点：
  - L 和 J 的梯度方向一致 ✅
- 远离当前点：
  - L 会偏离 J ❌

---

👉 所以：

> **L 是 J 的局部一阶近似（在梯度意义上）**

---

# 🔥 5. 为什么不能直接优化 L(θ)

因为：

- L 仍然是非线性的
- ratio 会爆炸
- 远离 θ_old 后不再可信

---

👉 所以再做一次近似：

---

# 🧩 6. 第二次近似：泰勒展开

在 θ_old 附近：

$$
L(\theta + \Delta \theta)
\approx
L(\theta) + g^T \Delta \theta
$$

其中：

$$
g = \nabla_\theta L(\theta_{old})
$$

---

👉 所以优化问题变成：

$$
\max_{\Delta \theta} g^T \Delta \theta
$$

---

# 💡 7. KL约束的引入（核心原因）

---

## ❗为什么必须限制？

因为：

- 上述近似 **只在局部成立**
- θ 变化太大 → L ≠ J → gradient 错误

---

## ✅ 所以引入：

$$
\mathbb{E}[KL(\pi_{old} || \pi_\theta)] \le \delta
$$

---

👉 二阶展开：

$$
KL \approx \frac{1}{2} \Delta \theta^T H \Delta \theta
$$

---

👉 最终优化问题：

$$
\max_{\Delta \theta} g^T \Delta \theta
\quad
\text{s.t. }
\Delta \theta^T H \Delta \theta \le \delta
$$

---

# 🧠 8. 参数空间 vs 策略空间（最关键理解）

---

## ❌ 错误直觉：

> 参数变化小 ⇒ 策略变化小

---

## ✅ 实际情况：

> 参数 → 策略 是非线性映射

---

👉 所以：

- 同样大小的 Δθ：
  - 在某些方向 → KL 巨大（敏感）
  - 在某些方向 → KL 很小（不敏感）

---

## 🎯 结论：

> TRPO 不限制参数变化，而是限制策略分布变化

---

# 🔥 9. KL约束的本质作用

---

## 🎯 核心作用：

> **保证 surrogate objective 的局部有效性**

---

## 🧠 因果链条：
KL约束
    ↓
限制策略变化只能很小
    ↓
surrogate近似成立
    ↓
梯度方向正确
    ↓
训练稳定
    ↓
旧数据可复用（副作用，bonus）



## 什么时候重新进行采样？参数$\theta$更新几次再去采样？
TRPO 是“保守的一步大更新”，  
PPO 是“多步小更新”