<!-- kar-module: riichienv.kar.kan_transactions -->
<!-- api: pending_kan_commits -->
# kan_transactions — 当前杠的公开提交判据

源码：[kan_transactions.py](../../../src/riichienv/kar/kan_transactions.py) · [总索引](../README.md)

## 职责与接口

`pending_kan_commits(event, actor=..., committed_kans=..., indicators=...) -> bool` 是纯公共事实适配器；判据由 [Rust observer::pending_kan_commits](../../../riichienv-core/src/observer.rs) 持有，Rust 观察者和重放上下文直接调用，Python tokenizer/独立结算经绑定调用。各层共享判据，Python 不再保留另一份杠接受公式。

合同版本为 `public_kan_commit_evidence_v2`。`COMMITTED_KAN_KINDS` 仅计已经成立的暗杠、加杠和大明杠，不把 pending 声明当完成副露。

## 判据

- 当前 actor 的补牌 `tsumo` 或明确 `kan_commit` 可以证明提交；actor 不匹配直接报错。
- 初始宝牌指示牌占一个位置。已成立杠仍有待揭示指示牌时，接下来的 dora 可能属于旧杠，不能提交新的 pending 杠。
- 只有已揭示数量已经对应全部旧杠时，新的 dora 才能证明当前杠；计数负数、零初始牌或多揭示等不一致拒绝处理。
- 其他事件返回 false；抢杠和流局的撤销由调用层管理。

```python
from riichienv.kar.kan_transactions import pending_kan_commits

# 已有一杠但只见初始指示牌：下一张 dora 仍可能是旧杠的。
commits = pending_kan_commits({"type": "dora"}, actor=2, committed_kans=1, indicators=1)
assert commits is False
```

## 调用关系与状态

不读牌山、暗手、他家合法动作或真实赢家，不修改环境，也没有独立 CLI。见 [observation](../observation/README.md)、[native_replay](../native_replay/README.md)、[tokenize](../tokenize/README.md)、[settlement](../settlement/README.md)。对应 [test_review_kan_transactions.py](../../../tests/kar/test_review_kan_transactions.py)。本轮未执行示例或规则对局。
