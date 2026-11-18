<!-- kar-module: riichienv.kar.event_semantics -->
<!-- api: PublicEventClock, public_history_positions, nine_terminals_public_hand -->
# event_semantics — 公共语义时钟与终局公开事实

源码：[event_semantics.py](../../../src/riichienv/kar/event_semantics.py) · [主设计 §6/8](../../KAR-v1.2.2-DESIGN.md) · [总索引](../README.md)

## 接口与合同

`PublicEventClock.advance(event)` 是只读过去事件的流式时钟。普通规范化公共事件各推进一个tick，展开多条Token共位置；source record编号和native batch不参与计算。同一触发牌连续多荣和共享裁定时间。可选kan_commit transport ACK不占模型时钟；当前账本仍可消费它的有效状态证据，历史接受行统一锚定共同的有效Dora/replacement draw，使旧杠延迟Dora和有无ACK不改变历史。SELF_PASS不推进公共时间。

`public_history_positions(events)` 在没有显式位置数组时，用相同合同处理已投影事件；不能使用旧 record 分组或 Token 行数当作逻辑时间。

`nine_terminals_public_hand(actor, hand, current_draw_actor=..., draw_count=..., has_discarded=..., calls_seen=..., source_tiles=...)` 只验证并返回宣布者的公开手牌：当前首次 draw、未舍牌未鸣牌、14张且至少9种幺九字、无第五张，源 tiles 若存在必须一致。调用方只能在已发生的九种九牌裁定后揭示，不能据此提前广播私有手牌或揭示其他玩家。

## 失败与集成

离线 `mjsoul` 和在线 `live_events` 共同使用九种九牌验证；所有账本通过 `PublicEventClock` 计时。不读取未来、他家隐藏状态或原始私有操作。缺少公开依据时不可伪装 visibility_complete；矛盾牌张或宣布者直接拒绝。

对应 [输入语义回归](../../../tests/kar/test_input_semantics.py) 与 [Small forward 对拍](../../../tests/kar/m1/test_input_semantics_forward.py)。本轮未执行真实数据重建、反向/参数更新、真实校准或独立棋力评测。这里规定合同，实际单元运行结果见[本轮报告](../../kar-semantic-branch-repair.md)，不代表训练阶段通过。
