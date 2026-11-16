<!-- kar-module: riichienv.kar.acceptance -->
<!-- api: consolidate, consolidate_shards -->
# acceptance — 来源绑定的最终数据 gate

源码：[acceptance.py](../../../src/riichienv/kar/acceptance.py) · [总索引](../README.md)

## 职责与接口

- `consolidate(main_dir, input_manifest, golden_manifests, output_dir)`：保留单 corpus 接口。
- `consolidate_shards(shards, golden_manifests, output_dir)`：合并多个 `(corpus_dir, input_manifest)` 对。

先重新检查原始字节 hash、完整 checkpoint、自洽的 report/receipt 与冻结合同，再全局做 UUID/raw/规范化语义去重和 Golden 别名排除。不同分片必须具有一致协议、实现、规则与 split salt。

## CLI

单片使用 `accept-data --corpus ... --input-manifest ... --output ...`；多片使用可重复的 `--shard CORPUS INPUT_MANIFEST`，可追加多个 `--golden-manifest`。单片参数与分片列表不可混用。

> 这些参数是说明，不是执行许可。汇总真实 Golden 会重新读取 raw 并执行当前规则重放，不能当作纯 JSON 拼接。

## 产物与不可替代的证据

在新输出目录保存标准 train/validation/test manifests、Golden manifest 与 `audit.json`。完整主数据 replay 消费已有来源绑定 receipt，不声称重新执行全部主样本；真实 Golden 则重新核验，报告区分两者并保留 producer 来源链。

不能用 Golden 补齐来豁免任一选定主游戏的 FAIL/BLOCKED。所有已保留 Golden、显式 golden_only 以及语义别名整场排除。支持计数按独立合格 train 游戏；N_train 决定模型阶梯。

多分片要求生产者提供 `result_event_semantics_v1` 摘要；旧片缺该证据时不能冒充已完成语义去重。输出不得覆盖 raw 或 producer 目录。

## 错误、测试与状态

源变动、身份冲突、报告关键字段与 checkpoint 不一致、缺 receipt 或冻结合同不同会拒绝。CLI 按 PASS/BLOCKED/FAIL 返回 0/2/1。对应 [test_acceptance.py](../../../tests/kar/test_acceptance.py)、[test_acceptance_shards.py](../../../tests/kar/test_acceptance_shards.py)。本轮未运行汇总或真实 Golden。
