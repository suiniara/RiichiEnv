<!-- kar-module: riichienv.kar.golden -->
<!-- api: find_golden_candidates, verify_golden_evidence, golden_coverage -->
# golden — 真实案例的发现、核验与覆盖

源码：[golden.py](../../../src/riichienv/kar/golden.py) · [注册表](../assets/README.md) · [总索引](../README.md)

## 职责与接口

`find_golden_candidates(game)` 从规范化事实定位可能案例；`verify_golden_evidence(game, replay_result, evidence)` 将具体案例与冻结期望、来源和完整 replay 结果核对；`golden_coverage(evidences)` 按注册表汇总必需真实正例及基于真实基线的派生检查。

候选定位、曾保存的 PASS 和 synthetic 单元测试都不是已核验的真实 Golden。

## 来源和失败边界

证据需要真实 game/raw/protocol/案例位置及对应期望。覆盖核验会重读来源并重新执行当前原生合法性检查，不能只信传入的历史状态字符串。未知里宝、支付链缺失或不符合冻结案例语义时拒绝，不从期望输出反推、修复事实。

当前证据版本 `mjsoul_real_golden_v2` 同时绑定顶层 `protocol_sha256` 和完整 `protocol_descriptor_sha256`。核验从证据登记的 `proto_path` 重新编译全部 imports，并在再次解码、重放前检查完整描述符摘要。修改依赖文件会使旧证据失败，即使顶层 proto 完全未变；缺少双重身份的 v1 证据必须重新产生，不补签未知的依赖身份。

真实 Golden 游戏整场保留，不能进入训练/验证/test 或独立 train 支持统计；即使没有成为某类的首份证据，显式 golden_only 也不能混入。规则可发生的稀有案例缺失继续 BLOCKED，不能改为 N/A。

## 使用与产物

独立 CLI 为 `python -m riichienv.kar.golden`，接收可重复 `--manifest`；完整其他参数见源码 `main()`。通常由 [corpus_audit](../corpus_audit/README.md)生成候选/来源，由 [acceptance](../acceptance/README.md)统一 gate。

输出说明已覆盖案例、缺失/错误项及保留游戏身份。它不是棋力评估，不启动模型训练，但真实核验仍会执行原生规则逻辑，需与只读静态文档工作分开。

## 测试与状态

对应 [test_golden.py](../../../tests/kar/test_golden.py)和各规则模块回归。本轮未读取真实案例、重跑 Golden 或更新历史验收结论。
