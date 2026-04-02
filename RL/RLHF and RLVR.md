👉 RLHF 存在 reward model 不准确的问题，因此通常不能进行大规模或长期优化，否则会出现 reward hacking。

👉 RLVR 通过使用可验证的客观 reward（如正确/错误），在 certain tasks（如数学、代码）上提供更稳定的优化信号，减少 reward model 带来的偏差。

👉 但 RLVR 仍然面临 sparse reward、credit assignment 和 exploration 等问题，且仍可能出现 specification gaming，因此也不能无限制训练。

👉 当前最有效的范式是 SFT + RLHF + RLVR 的组合，而不是单一方法。