<!-- kar-module: riichienv.kar.agent -->
<!-- api: StaticLogitPredictor, KARAgent, run_match, functional_match -->
# agent — 在线选择与完整半庄执行

源码：[agent.py](../../../src/riichienv/kar/agent.py) · [总索引](../README.md)

## 主要接口

- `KARAgent(seat, policy)`：绑定稳定观察者 ID 和 `RuntimePolicy`。
- `choose(observation, snapshot, round_instance, bounds=None)`：校验本家上下文，映射完整合法集合，采样并返回 `MappedAction` 与 `PolicyResult`。
- `run_match(agents, wall_seed, max_decisions=10000)`：四个显式分配的 agent 完整执行冻结规则半庄。
- `StaticLogitPredictor` / `functional_match(...)`：模型无关的集成 fixture，不代表训练策略或棋力。

## 执行合同

own-action、两步立直、所有活跃响应者的联合提交和自动推进分别处理。合法可抢暗杠的空原生日志窗口由已验证动作和公开阶段推进，而不是误判停滞。

策略只取得本家 Observation 和白名单快照；不能依赖全知环境筛选 response mask。响应选择先捕获 owner-only Pass 上下文；原子执行失败/取消不确认，成功后才向 [live_events](../live_events/README.md) 提交私有确认。它只进入本家后续历史，动作前输入和公共 trace 不变。

## 输出与边界

完整结果含 scores/ranks、事件、每次决策/延迟/telemetry、墙 seed 和最终策略 RNG 摘要。逻辑 match hash 不把运行耗时当确定性游戏事实。触达决策 cap 返回 BLOCKED，不能计作完成半庄。

该 runner 的单局 PASS 仅证明完整执行，不自动给出棋力统计；正式成对规模、对手、预算和区间由 [learning/simulation](../learning/simulation/README.md)约束。`choose(bounds=...)` 可消费外部有效界，runner 不提供统一误差估计器。

## 使用与状态

`python -m riichienv.kar.agent` 是固定静态 logits 的功能入口，包含实际规则对局，不是纯帮助/静态检查。没有账号连接、在线雀魂客户端或 MCTS。

对应 [test_self_pass_history.py](../../../tests/kar/test_self_pass_history.py) 和 [core/test_live_agent.py](../../../tests/kar/core/test_live_agent.py)。本轮未运行完整半庄或阶段任务；只用真实原生环境与受控策略，设8次决策 cap 的合成回归验证 Pass 接线及失败不确认。
