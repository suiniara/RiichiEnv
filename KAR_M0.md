# KAR M0 — RiichiEnv 原生数据与规则验收

本目录实现 **CPU 数据/规则审计**，不是训练好的模型，不包含 M1 网络、GPU 训练或 M3 评估。RiichiEnv 是实现宿主；Kanachan 只作为 546 动作编号的冻结参考，不是运行依赖。原始设计包不作修改。

## 冻结协议

- 活动 Schema：`kar-riichienv-v1`。
- 完整定义：`src/riichienv/kar/token_schema.riichienv.v1.json`。
- Canonical JSON SHA-256：`fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673`。
- 参考设计仍为 `kar-v1.2.2`；overlay 的字节 hash 不是完整 Schema hash。
- 不声称恢复了缺失的 v1.2.1 kind registry，也不声称新版字节与旧 registry 兼容。
- 保留 944 行输入 Embedding、48 kind、546 动作、32 候选、62 状态槽。
- 单 Token 为 139 字节 little-endian；历史可见事实按明确规则分块，H 为分块后的历史行数。前缀 H+64 行，追加 ACTION 后 H+65 行。
- `TokenHistoryCache` 是 CPU 序列化缓存，不是模型 KV cache。观察者、初始顺位、Schema hash 或历史前缀改变会使缓存失效。封存的行深冻结，不允许修改行后继续输出陈旧字节。

## 数据与规则

主训练/验证/测试 cohort 固定为雀魂玉之间四人半庄：`category=2`、`mode=2`、`mode_id=12`、四个稳定座位。其他同规则房间只能提供规则金样本，不进入主训练统计。天凤不混入。

- 冻结 `GameRule.default_mjsoul()`，`sanchaho_is_draw=False`。
- 三家和流局不适用；三家荣和 `RON_MASK_7` 保留。
- 允许国士无双抢暗杠；起家顺东/南/西/北破同分。
- 实际 Protobuf 动态描述符来自指定 `.proto`；生成的 pb2 仅允许静态 AST descriptor 对比，不执行下载的 Python。
- version 210715 的 actions 只读取 `.result`；`.user_input` 是不透明跳过字段。
- 原始字节、规范化事件、协议、规则、Schema 和动作 manifest 各自保留 hash，不互相冒充。

## 独立结算与完整性

`apply_event()` 用于状态重放，不把它的分数当成独立结算证据。`HandEvaluator` 与独立分数/供托账本从事件事实计算番符和付款；服务器的 han/fu/fans/payment 标签只是比较目标，不能反过来改变计算。

流局省略 delta 时保持“标签缺失”，不自动填零。可由独立计算的分数与下一局公开起始分比较；真正达到独立半庄终止条件时，也可与按座位的 header 原始分比较。后者只是款项比较，不能单独证明半庄完整，不能恢复缺里宝等未知计算链。

旧牌谱没有显式 `end_game` 时，只有完整初局、全事件消费、合法动作/响应、独立款项、冻结终止规则及实际 header/最后记录证据全部满足，才签发派生终局证书。**不会补造原始结束事件**，报告保留 `explicit_end_game=false`。

杠响应由只读原生薄接口在 GameState 副本上预览，返回完整 RON/PASS 集合；不按真实赢家筛选，不把未提交加杠写入观察者输入，不返回模拟摸牌。新增上下文桥只补前缀可推导的一发、双立直、岭上及漏和标记，不改 Rust Core 规则。

## Golden 与 gate

`golden_registry.v1.json` 区分 16 类真实正例与 11 类基于真实基线的派生检查。候选、合成单元测试、保存的 PASS 标签均不等于真实 Golden 通过。

验证会重新打开原始字节并解码，核对期望输出、独立结算和终局证书；覆盖汇总还会重新执行当前原生合法性重放，每份 raw/protocol 组合只重跑一次。缺少必需真实案例继续 BLOCKED，规则可发生的稀有案例不能标 N/A。

选入 Golden manifest 的游戏整场保留为 `golden_only`，从主训练清单及响应支持统计中排除。

- `PASS` / exit 0：当前明确输入范围内的验收证据满足。
- `BLOCKED` / exit 2：证据不足、缺实际场景或没有可用 train 游戏，不代表程序崩溃。
- `FAIL` / exit 1：已发现数据或合同错误。
- 一份游戏通过不代表全盘通过；有未解决游戏不能静默过滤后宣称整批 PASS。
- 规模只按 `N_train_half_games`：少于 30k Small，30k–49,999 Medium，50k 及以上 Large。
- 响应分支支持只计独立 train 游戏，至少 500；validation/test 不补数量。支持数量不是一致误差保证。

## 运行与报告

在 RiichiEnv 工程内运行，使用其现有 uv 环境。下面路径仅为本机工作区示例；输出目录应为新目录，且必须在原始输入目录之外。不要在下载目录里运行解释器或导入其中脚本。

单份/少量显式 raw 审计：

```bash
uv run --no-sync python -m riichienv.kar.audit --proto ../kar-m0-assets/mahjongsoul.proto --record ../kar-m0-assets/200331-00a872c6-afb1-487b-ba2d-e58b9379f60e.bin --record-provenance real_raw_record --split-salt kar-riichienv-v1-games --output ../kar-m0-reports/single-audit.json
```

提供已生成的真实金样本清单时，增加 `--golden-manifest <path>`。主 cohort 不允许通过命令行改成其他房间。旧 mode9 样本可以核验规则和终局，但不会计入玉之间训练数量。

固定 manifest 的有界批量审计：

```bash
uv run --no-sync python -m riichienv.kar.corpus_audit --manifest ../kar-m0-assets/jade-catalog-1000/manifest.jsonl --proto ../kar-m0-assets/mahjongsoul.proto --output ../kar-m0-reports/new-corpus-audit --mode audit --limit 1000 --split-salt kar-riichienv-v1-games
```

中断后用完全相同参数并增加 `--resume`。源字节、输入清单、协议/规则/Schema、实现代码指纹、split salt 或已完成回放 receipt 不一致时拒绝沿用旧结果。重复视角不增加游戏数；同 ID 不同 hash 是冲突，不静默取第一份。

`--mode discover` 只定位场景，永远不产生训练资格。`corpus_fetch` 只是只读搬运器，旧 catalog 仅作定位，不继承旧 split 或 PASS 结论。扫描统计仅覆盖明确、有界输入；`all_disk_scanned` 保持 false。

批量输出：

- `checkpoint.json`：输入范围、实现指纹、逐游戏进度与完整性摘要。
- `receipts/<game_id>.json`：本项目的当前完整回放结果。
- `train.manifest.json` / `validation.manifest.json` / `test.manifest.json`。
- `golden.manifest.json`：可重新验证的来源绑定证据。
- `report.json`：实际处理/通过/失败/阻断数、真实覆盖缺项、N_train 与 train-only 分支支持。

分阶段结果可用多个真实 Golden manifest 做最终 gate 汇总；不会覆盖原来的 scoped 报告，也不会把其中的失败/阻断游戏过滤掉：

```bash
uv run --no-sync python -m riichienv.kar.acceptance --corpus ../kar-m0-reports/audit-jade-1000-v2 --input-manifest ../kar-m0-assets/jade-catalog-1000/manifest.jsonl --golden-manifest ../kar-m0-reports/audit-jade-1000-v2/golden.manifest.json --golden-manifest ../kar-m0-reports/rare-20-audit/golden.manifest.json --output ../kar-m0-reports/final-acceptance
```

最终汇总重验原始文件 hash、完整 checkpoint、逐游戏 receipt、冻结协议和当前真实 Golden。主数据原生重放使用来源绑定的已完成 receipt，不声称再次执行整批；报告保留 producer 实现指纹及原报告状态。任何未处理输入、缺 receipt、源字节变化、未通过游戏或不足的 Golden 都不能得到最终 PASS。显式 `golden_only` 来源即使没有被选作某类的首个证据，也不能进入训练。

有界审计分片可由 `acceptance.consolidate_shards()` 或重复 `--shard CORPUS INPUT_MANIFEST` 汇总，不取消单片 10000 输入上限。合并器检查全部分片的协议、实现、规则和 split salt，核验 raw/checkpoint/receipt，对游戏 ID、原始字节与新生产者保存的规范化语义进行全局去重，并把所有 Golden 的语义别名一并排除。输出保持标准 split manifests，完整导出可据此达到 30k／50k train-game 模型门槛。

多分片要求 `result_event_semantics_v1` 来源绑定摘要；旧分片缺该摘要时不会被静默当作已完成语义去重。单 corpus 的旧来源证据接口继续保留，原报告状态和原始文件不作覆盖。

所有读原始数据的辅助 Python 命令应使用 `-I`；正式模块从工程环境运行，不执行旧数据工作区代码。默认一个审计进程；未获授权不全盘遍历、不训练、不启动付费资源。

测试与静态检查：

```bash
uv run --no-sync pytest -q tests/kar
```

```bash
uv run --no-sync pytest -q
```

```bash
uv run --no-sync ruff check .
```

```bash
uv run --no-sync ty check
```

私有 raw 不随源码分发；没有资产时，依赖它们的集成测试会明确 skip，不能据此声称真实覆盖通过。
