<!-- kar-module: riichienv.kar.actions -->
<!-- api: ActionSpec, MappedAction, ActionExecution, AmbiguousDiscardError, manifest, manifest_hash, action_spec, action_id, legal_candidates, execute_candidate, execute_joint_responses, tile136_to_tile37, tile37_to_base34 -->
# actions — 原生动作与 546 全局 ID

源码：[actions.py](../../../src/riichienv/kar/actions.py) · [总索引](../README.md)

## 职责

维护完整动作语义 manifest，将本家原生合法 `Action` 转成全局候选，再把选定候选安全执行回 RiichiEnv。原生 action index、候选数组位置和全局 ID 是不同东西。

## 主要接口

| 接口 | 用途 |
|---|---|
| `manifest()` / `manifest_hash()` | 全部 546 个冻结 `ActionSpec` 及身份 |
| `action_spec(id)` / `action_id(spec)` | 完整语义的双向查询 |
| `legal_candidates(observation, ...)` | 本家完整合法集合 → `MappedAction`；保留原生签名 |
| `execute_candidate(env, player_id, mapped)` | 执行 own-action；立直可能包含 reach/dahai 两步 |
| `execute_joint_responses(env, choices)` | 校验所有活跃响应者后一次联合提交 |
| `ActionExecution` | 保留所有 step observations，而非只返回最后一个阶段 |
| `tile136_to_tile37` / `tile37_to_base34` | 显式转换牌编码，不能直接互换整数域 |

## 合同与使用

摸切/手切、立直与否、普通五/赤五分别保留完整动作身份；容量为 32，不丢变体或 Top-K。立直可打牌的枚举不能因可见四张而排除结构听牌的空听。来源玩家应从公开舍牌或 pending kan 推导，不能用真实赢家反推候选。

```python
from riichienv.kar.actions import manifest, action_spec

assert len(manifest()) == 546
spec = action_spec(0)
```

执行会推进真实环境，不能把执行后状态写回该动作的决策前输入。多人响应先验证全部选择，不能依次 step 而丢失其余荣和者。

## 错误与调用关系

非法 ID、原生签名过期、候选与环境不一致、响应者集合不完整或不可安全消歧会报错；`AmbiguousDiscardError` 不应通过过滤概率来隐藏。上游是观察者/原生 Observation，下游是 [agent](../agent/README.md)、[audit](../audit/README.md)及导出。

## 测试与状态

对应 [test_actions.py](../../../tests/kar/test_actions.py)、[test_kan_preview.py](../../../tests/kar/test_kan_preview.py)、[core/test_live_agent.py](../../../tests/kar/core/test_live_agent.py)。本轮未执行示例、原生动作或测试。
