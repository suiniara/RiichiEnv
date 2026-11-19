<!-- kar-module: riichienv.kar.completion -->
<!-- api: observed_raw_final_scores, certify_completion -->
# completion — 完整半庄终局证书

源码：[completion.py](../../../src/riichienv/kar/completion.py) · [总索引](../README.md)

## 职责与接口

`observed_raw_final_scores(game)` 读取按稳定座位对应的原始最终分数作为比较目标。`certify_completion(game, replay_result)` 将完整 replay、独立结算、冻结终止规则和 raw 结束证据结合，判定是否可用于完整半庄监督与最终名次。Python 验证最后一局身份、赢家组或庄家听牌证据，再调用 `native_round_transition(..., rule=MJSOUL_RULESET.engine_rule())`；终止规则与采用雀魂配置的在线状态机及 `settlement` 共用 [Rust 内核](../../../riichienv-core/src/rules/settlement.rs)，不再另算一份终止条件。

它不把原生 `ranks()`、最后一条服务器排名标签、分数相等或共享内核的 `match_end` 单独当作完整性证书。来源覆盖、结束证据和 FAIL/BLOCKED 分类仍属于 Python 审计职责。

## 证据合同

当前 `frozen_mjsoul_hanchan_completion_v2` 证书包含 `protocol_identity_version`、顶层源协议 `protocol_sha256` 和含 imports 的 `protocol_descriptor_sha256`。缺少完整描述符身份会 BLOCKED；仅依赖协议发生变化，也会产生不同证书摘要。来源身份由解码器传入，不能用旧证书字段重新签名替代重放。

旧牌谱可能没有显式 `end_game`。仅当初局/回合链完整、全部事件消费、合法动作/响应、独立款项和实际终局条件都满足，才能签发派生终局结果；保留 `explicit_end_game=false`，不补造原始事件。

固定起家东/南/西/北破同分，当前 `oya` 不改变映射。header 原始分只能用于核对，不能替代缺失的独立计分链；未知里宝或未消费事件不能靠 header 排名修复。KAR 雀魂西入普通荒牌达标时结束，途中流局及庄家双荣和等已有例外单独处理；天凤的普遍连庄优先不能沿用到此证书。两平台由 `extension_dealer_repeat_priority` 区分，见[官方来源与差异表](../../kar-native-rule-boundary.md#3-平台差异与-kar-雀魂修复)。

## 输入、输出与调用方

输入是经过 [mjsoul](../mjsoul/README.md)规范化的游戏和 [audit](../audit/README.md)真实 replay 结果。输出为带条件/原因/比较证据的完成状态；`learning/export` 只对满足完整性合同的数据生成最终名次监督。

无独立 CLI，不训练模型，不写原始文件。缺必需起局/结束事实为证据不足；与规则或账本矛盾为错误。不能把截断或短局直接当完整半庄。

## 测试与状态

对应 [test_completion.py](../../../tests/kar/test_completion.py)、[test_draw_obligations.py](../../../tests/kar/test_draw_obligations.py)、[test_audit.py](../../../tests/kar/test_audit.py)、[test_native_settlement_kernel.py](../../../tests/kar/test_native_settlement_kernel.py)。实际验证范围见[边界迁移记录](../../kar-native-rule-boundary.md)；历史证书必须重新核对原始数据、当前规则、结算和源码/原生构建绑定，不能重贴标签冒充新证书。
