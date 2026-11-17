<!-- kar-module: riichienv.kar.audit -->
<!-- api: check_overlay, compare_generated_descriptor, replay_game, run_audit, response_support_evidence -->
# audit — 单游戏完整重放与 M0 协调

源码：[audit.py](../../../src/riichienv/kar/audit.py) · [总索引](../README.md)

## 职责与接口

`replay_game(game, on_decision=None)` 是数据链核心：根据真实规范化记录建立原生环境、四份观察者账本、合法动作/响应、输入编码检查和独立结算，再核验半庄完整性。可通过 `on_decision` 把经过截止点检查的 `ReplayDecision` 交给导出器。

`run_audit(...)` 汇总显式 raw 输入、Schema/规则/cohort/Golden 等检查。`check_overlay` 不把 overlay 当完整 Schema；`compare_generated_descriptor` 只静态对比生成模块中的 descriptor，不导入执行下载的 pb2。`response_support_evidence` 将事件独立游戏数按本项目 game split 统计。

## 数据边界

监督处理可以查看后续事实作标签，但当前决策前缀只来自冻结 cutoff。唯一正常 NEXT_DRAW 产生 Pass 监督时，私有 SELF_PASS 仍等到循环实际处理该确认摸牌后才写入本家后续历史；有无 export callback 都得到相同后续输入。被覆盖的 CHI/PON/争议意图不伪造为 Pass，provenance 保留。原生阶段和独立结算检查继续执行。

服务器番符/支付和 header 分数仅作比较目标，不得反过来修正独立账本。缺少响应、里宝或完整结算证据时保持未知/BLOCKED，不能填零。相同 game 的四视角不增加独立支持数。

## 使用与输出

独立 CLI 提供 `--proto`、可重复 `--record`、`--container`、`--record-provenance`、Schema/Golden 等参数；完整参数以源码 `main()` 为准。正常生产批量由 `corpus_audit` 调用，不要把一次显式文件审计声称为全盘验收。

结果含阶段、输入编码、独立结算、终局等组件状态，及事件/响应计数。`training_allowed` 只表示该审计范围的数据 gate，不等于用户许可。

## 测试与状态

对应 [test_self_pass_history.py](../../../tests/kar/test_self_pass_history.py)、[test_audit.py](../../../tests/kar/test_audit.py)、[test_native_replay.py](../../../tests/kar/test_native_replay.py)。本轮未运行阶段任务或真实校准/重导出；仅有界输入链路与桥接回归，完整半庄证书不由这些测试替代。
