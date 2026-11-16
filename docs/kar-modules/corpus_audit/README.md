<!-- kar-module: riichienv.kar.corpus_audit -->
<!-- api: normalized_game_sha256, audit_corpus, select_golden_inputs -->
# corpus_audit — 有界主数据审计与恢复

源码：[corpus_audit.py](../../../src/riichienv/kar/corpus_audit.py) · [总索引](../README.md)

## 职责与接口

`audit_corpus(manifest_path, proto_path, output_dir, ...)` 从明确的搬运清单中逐游戏解码、重放、编码与独立核验，保存 scope 和每游戏 receipt。`normalized_game_sha256(events)` 保存 `result_event_semantics_v1` 语义摘要；`select_golden_inputs(...)` 从 discovery 中选择单独的真实候选来源。

## discover 不等于 audit

CLI：`python -m riichienv.kar.corpus_audit --manifest ... --proto ... --output ...`。必须注意 `--mode` 缺省是 `discover`，只有显式 `--mode audit` 才进行完整审计；还可使用 `--limit --split-salt --resume`。

每片限制 1..10000 输入，输出目录必须独立。`discover` 只找候选，不产生训练许可。不能通过跳过坏游戏把所选整批报告为 PASS。

## 产物与身份

- `checkpoint.json`：明确输入范围、完整进度及实现指纹。
- `receipts/<game_id>.json`：绑定 raw/protocol 的完整 replay 证据。
- `report.json`、split manifests、`golden.manifest.json`：范围结果与可复核证据。
- 语义摘要忽略来源坐标及 start_game 身份字段，但保留游戏事实，用于后续跨分片别名去重；不替代原始字节 hash。

resume 要求原始来源、协议、规则、Schema、实现、split salt 等一致。单片上限不阻断 Medium/Large：使用 [acceptance](../acceptance/README.md)合并多个已完成分片，而不是去掉安全上限。

## 调用关系与失败处理

上游 `corpus_fetch`，核心调用 `mjsoul → audit.replay_game`，下游 `acceptance → learning/export`。FAIL 是已发现错误，BLOCKED 是证据未完成；二者都不能授予训练资格。Golden 整场排除，独立支持只统计最终合格 train 游戏。

## 测试与状态

对应 [test_corpus_audit.py](../../../tests/kar/test_corpus_audit.py)、[test_acceptance_shards.py](../../../tests/kar/test_acceptance_shards.py)。本轮未读取原始数据、执行审计或覆盖历史报告。
