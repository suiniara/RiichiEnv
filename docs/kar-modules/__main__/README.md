<!-- kar-module: riichienv.kar.__main__ -->
<!-- api: main -->
# __main__ — 顶层延迟导入命令

源码：[__main__.py](../../../src/riichienv/kar/__main__.py) · [总索引](../README.md)

## 职责

提供 `python -m riichienv.kar COMMAND ...`。`main()` 从 `COMMANDS` 查到目标模块及固定前缀，临时构造 argv、延迟导入并转发到目标 `main()`，最后恢复原 argv。帮助输出不需要预先导入所有 ML 模块。

## 命令去向

| 命令 | 对应说明 |
|---|---|
| `accept-data` | [acceptance](../acceptance/README.md) |
| `export-data` | [learning/export](../learning/export/README.md) |
| `train` | [learning/trainer](../learning/trainer/README.md) |
| `m1` / `m2` | [learning/orchestration](../learning/orchestration/README.md) |
| `plan-benchmark` / `plan-m3` | [planning](../planning/README.md) |
| `calibrate` / `select-tau` / `release` / `benchmark` / `m3` | [learning/jobs_cli](../learning/jobs_cli/README.md) |
| `stamp-build` | [release](../release/README.md) |

## 使用与副作用

```bash
uv run --no-sync python -m riichienv.kar --help
```

命令参数由目标模块解释；本分发层不统一抹平各模块错误，不创建授权、不自动修复输入或改变预算。未知命令直接拒绝。`plan-*` 为元数据工具；其他命令可能读取大量数据、写产物、更新参数或运行模型，必须阅读目标文档。

有些 M0/历史诊断模块通过各自 `python -m riichienv.kar.<module>` 进入，不在此别名表中。不要把历史 CPU smoke 当成 `train` 的正式实现。

## 测试与状态

入口默认拒绝计算的覆盖见 [production/test_jobs.py](../../../tests/kar/production/test_jobs.py)，元数据计划见 [production/test_metadata_planning.py](../../../tests/kar/production/test_metadata_planning.py)。本轮未执行帮助命令、计算入口或测试。
