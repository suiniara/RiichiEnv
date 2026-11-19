<!-- kar-module: riichienv.kar.payments -->
<!-- api: EvaluatedWin, win_payments, draw_payments, end_game_pot -->
# payments — 共享原生付款与供托结算适配

源码：[payments.py](../../../src/riichienv/kar/payments.py)、[Rust settlement](../../../riichienv-core/src/rules/settlement.rs)、[绑定](../../../riichienv-python/src/rules.rs) · [总索引](../README.md)

## 职责与接口

由已独立评估的和牌事实计算四家点数变化，不把服务器 payment/delta 当计算输入。Python 负责类型、座位、计数和结果对象适配；支付公式由 Rust `rules::settlement` 持有，四人在线自摸、整组荣和、流局和终局供托调用同一内核。本 KAR 适配层固定雀魂；原生通用状态机按其 `GameRule` 保留平台差异，共享内核不表示天凤与雀魂必须采用同一规则。

| 接口 | 输入 → 输出 |
|---|---|
| `EvaluatedWin` | 赢家、独立 `WinResult` 与适用责任支付事实 |
| `win_payments(wins, oya, target, honba, riichi_sticks)` | 自摸/一至三家荣和 → 四家 delta tuple |
| `draw_payments(tenpai, oya, nagashi_winners=(), abortive=False)` | 四家听牌及流局类型 → 四家 delta tuple |
| `end_game_pot(scores, riichi_sticks, initial_order)` | 终局残余供托处理 → 按冻结规则的结果 |

`target=None` 表示自摸且只能一位赢家；荣和目标不能同时是赢家。多人荣和供托与本场顺序按实现的冻结雀魂规则处理；不要使用其他平台或赛事的分配方式替换。

## 守恒与错误

普通和牌付款的 delta 总和等于投入赢家的供托；不含供托的流局支付和为零。责任支付役满单位、赢家唯一性、座位、本场和供托等输入均验证。非法事实应报错，不通过对差额“找补”让总分看似一致。

本层不枚举手牌役种或自行判听；那由 [settlement](../settlement/README.md)依据牌和冻结雀魂规则调用原生 `HandEvaluator`。听牌/等待查询须显式传规则，无参旧接口保留天凤兼容语义。审计输入要求零本场的基础评估，原生内核重新计算基础支付并只分配一次本场与供托。反过来，不能按 header/支付差额猜番符后再调用本模块自证。

这里的“独立”是独立于服务器支付标签，并非维护第二套 Python 规则公式。共享规则不替代来源证据与已知案例核验；三人支付仍走其原有实现，本轮不声称三人雀魂规则已迁移或修复。

## 使用与状态

库接口，无独立 CLI，不读取文件或创建优化器。审计结果把这里的计算与服务器标签/下一局公开分数比较。对应 [test_payments.py](../../../tests/kar/test_payments.py)、[test_settlement.py](../../../tests/kar/test_settlement.py)、[test_native_settlement_kernel.py](../../../tests/kar/test_native_settlement_kernel.py)。构建、测试与旧产物迁移要求见[原生规则边界](../../kar-native-rule-boundary.md)。
