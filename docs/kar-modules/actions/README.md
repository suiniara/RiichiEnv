<!-- kar-module: riichienv.kar.actions -->
<!-- api: ActionSpec, MappedAction, ActionExecution, AmbiguousDiscardError, manifest, manifest_hash, action_spec, action_id, legal_candidates, execute_candidate, execute_joint_responses, tile136_to_tile37, tile37_to_base34 -->
# actions — 原生动作与 546 全局 ID

源码：[actions.py](../../../src/riichienv/kar/actions.py) · [总索引](../README.md)

## 职责

维护完整动作语义 manifest，将本家原生合法 `Action` 转成全局候选，再把选定候选安全执行回 RiichiEnv。麻将合法性、完整立直可舍牌集合和立直原子提交属于 Rust；Python 不再为每个候选另做一次听牌判断。原生 action index、候选数组位置和全局 ID 是不同东西。

## 主要接口

| 接口 | 用途 |
|---|---|
| `manifest()` / `manifest_hash()` | 全部 546 个冻结 `ActionSpec` 及身份 |
| `action_spec(id)` / `action_id(spec)` | 完整语义的双向查询 |
| `legal_candidates(observation, ...)` | 本家完整合法集合 → `MappedAction`；保留原生签名 |
| `execute_candidate(env, player_id, mapped)` | 执行 own-action；完整立直通过原生 `step_riichi_discard` 原子提交 |
| `execute_joint_responses(env, choices)` | 校验所有活跃响应者后一次联合提交 |
| `ActionExecution` | 保留完整原子执行返回的阶段 observations |
| `tile136_to_tile37` / `tile37_to_base34` | 显式转换牌编码，不能直接互换整数域 |

## 合同与使用

摸切/手切、立直与否、普通五/赤五分别保留完整动作身份；容量为 32，不丢变体或 Top-K。完整立直可舍牌来自冻结本家 Observation 的 `legal_riichi_discards()`，使用该 Observation 保存的环境规则，与原生声明后舍牌检查共享判据，不因外部可见四张而排除结构听牌的空听。KAR 使用显式雀魂规则；通用天凤/default 的形式听牌与无参旧 `HandEvaluator` 语义不能被全局改写。来源玩家应从公开舍牌或 pending kan 推导，不能用真实赢家反推候选。

```python
from riichienv.kar.actions import manifest, action_spec

assert len(manifest()) == 546
spec = action_spec(0)
```

执行会推进真实环境，不能把执行后状态写回该动作的决策前输入。Python 仍用 reach/dahai 两个原生动作保存完整映射身份，但执行时把完整立直舍牌交给一个原生原子接口；失败不得留下已声明立直的半完成状态。多人响应先验证全部选择，不能依次 step 而丢失其余荣和者。

## 错误与调用关系

非法 ID、原生签名过期、候选与环境不一致、响应者集合不完整或不可安全消歧会报错；`AmbiguousDiscardError` 不应通过过滤概率来隐藏。缺少完整立直枚举或原子提交接口的旧扩展明确失败，不能退回 Python 规则判断或非原子的 reach/dahai 两次提交。上游是观察者/原生 Observation，下游是 [agent](../agent/README.md)、[audit](../audit/README.md)及导出。

## 测试与状态

对应 [test_native_rule_boundary.py](../../../tests/kar/test_native_rule_boundary.py)、[test_actions.py](../../../tests/kar/test_actions.py)、[test_kan_preview.py](../../../tests/kar/test_kan_preview.py)、[core/test_live_agent.py](../../../tests/kar/core/test_live_agent.py)。546 动作身份与模型形状不变；实际验证与旧产物迁移见[原生规则边界](../../kar-native-rule-boundary.md)。
