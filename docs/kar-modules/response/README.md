<!-- kar-module: riichienv.kar.response -->
<!-- api: ResponseResolution, draw_response, resolve_discard_response -->
# response — 实际舍牌响应窗口标签

源码：[response.py](../../../src/riichienv/kar/response.py) · [总索引](../README.md)

## 职责与接口

`resolve_discard_response(events, trigger_event_id, ruleset=MJSOUL_RULESET)` 从一个真实舍牌触发点检查后续窗口，返回 `ResponseResolution`。`draw_response(reason)` 将已知流局原因映射到冻结响应类。

此处是**离线标签解析**，允许核对窗口结束事实；不是模型的公开规则 mask，不得把其结果反向写入 cutoff 输入。

## 标签合同

`ResponseResolution` 区分 `response_applicable`、`response_observed`、target 和理由/窗口信息。非舍牌不适用，未知/缺日志不算正常 NEXT_DRAW。多人荣和合成完整相对座位 bit mask，三家和按冻结 ruleset 决定实际荣和或流局，不能只保留首名赢家。

P0 是本次舍牌者，响应相对座位 P1/P2/P3 不能直接当固定玩家 ID。吃碰大明杠与杠后岭上结果也不能混为同一个窗口事件：原舍牌被大明杠接走时，其标签仍是 DAIMINKAN。

## 使用与调用关系

通常由 `learning.export.GameWriter` 在已经完成 replay 的游戏上调用；`audit` 验证窗口完整性。输入必须包含真实、按序且未截断的事件上下文；不提供独立 CLI，不执行模型或动作。

下游用 [response_mask](../response_mask/README.md)验证有效 target 必须被公开 mask 允许。发现冲突应报数据合同错误，不为了容纳标签而 unmask。

## 测试与状态

对应 [test_observation_response.py](../../../tests/kar/test_observation_response.py)、[test_draw_obligations.py](../../../tests/kar/test_draw_obligations.py)、[m1/test_dataset.py](../../../tests/kar/m1/test_dataset.py)。本轮未执行标签解析或测试。
