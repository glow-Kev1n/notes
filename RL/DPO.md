DPO 不是在训练 reward model，而是在训练 policy，但这个 policy 的更新效果等价于隐式地满足了 reward preference 的要求。

##  DPO 为什么不要 reward model

因为 DPO 发现：

对于这个带 KL 正则的 RL 问题，最优策略满足

$$\pi^*(y|x)\propto \pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

把它改写一下：
$$r(x,y)=\beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta \log Z(x)$$
这一步很关键,DPO 里则把它替换成了一个由**最优**policy 定义的隐式 reward：
后续优化的时候，
- 随机参数化表示$\pi_\theta$，用上面这个公式去隐式地定义reward
- 再让这个隐式 reward 在偏好数据上符合 Bradley-Terry 模型
- 训练完成后，得到的$\pi_\theta$ 就希望接近那个理论上的 $\pi^*
它说明：

> **reward 和最优策略之间不是彼此独立的，reward 可以写成 policy 相对 reference 的 log-ratio。**
> 
> **DPO 的 loss 不是在训练一个独立的 reward model，而是在直接训练 policy；同时这个 policy 相对于 reference policy 的 log-ratio 可以被解释为一个隐式 reward，因此不再需要单独训练 reward model。**

所以原来流程是：
- 先学 rϕr_\phirϕ​
- 再根据 rϕr_\phirϕ​ 去学 π\piπ

而 DPO 说：

- 既然reward和最优策略有闭式关系
- 那我就没必要单独显式学一个$r_\phi$
- 直接学$\pi_\theta$​ 就行

因为偏好数据只关心：
$$y_w \succ y_l$$

传统方法会说：

“先学一个 reward，使 winner 分高于 loser。”

DPO则说：

“我直接让 policy 满足 winner 相对 ref 更被提升，loser 相对 ref 更被压低。”

具体就是让下面这个量变大：

$$\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$$

这其实就等价于在让一个隐式 reward difference 变大。

所以 reward model 的工作，被 policy-ratio 这套表达替代了。
## 绝对分数与人类偏好数据


## 传统RLHF与DPO

### 传统RM是怎么训练的？数据是什么？


## Bradley-Terry：

