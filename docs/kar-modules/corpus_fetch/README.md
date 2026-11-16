<!-- kar-module: riichienv.kar.corpus_fetch -->
<!-- api: CorpusFetchError, FetchConfig, validate_raw_path, ssh_command, fetch_corpus -->
# corpus_fetch — 有界只读数据搬运

源码：[corpus_fetch.py](../../../src/riichienv/kar/corpus_fetch.py) · [总索引](../README.md)

## 职责

从明确 SSH 主机和 SQLite locator 中选择有界 raw 记录，核验来源后写入新的本地输出。它是搬运器，不是 M0 规则/结算验收，也不继承旧 catalog 的 PASS 或训练 split。

## 主要接口

| 接口 | 作用 |
|---|---|
| `FetchConfig(host, ssh_executable, remote_python, catalog, ...)` | 冻结连接、选择范围与游标 |
| `FetchConfig.scope()` | 验证路径/host/limit，输出可恢复的 scope |
| `validate_raw_path(value, uuid)` | 对允许原始目录和身份做严格路径检查 |
| `ssh_command(config, wire)` | 构造指定解释器的受限远端只读命令 |
| `fetch_corpus(config, output_dir, resume=False)` | 流式搬运与检查点恢复 |

## 配置与产物

`limit` 为 1..1000；`mode_id` 只接受代码允许的 12/16，后者仅作为独立 Golden 来源。`source_split` 是旧 locator 的筛选字段，不是本项目的新 game split。remote Python 必须是通过验证的绝对 Windows 可执行路径，catalog 必须指向 SQLite 文件。

输出包含本地 raw、来源 manifest、checkpoint 和范围统计；已有来源/部分写入的恢复必须核对 scope、长度和 hash，不覆写冲突文件。`all_disk_statistics=false`，不得把这一批当全盘普查。

## 使用与副作用

独立 CLI 为 `python -m riichienv.kar.corpus_fetch`；参数包括 `--host --ssh-executable --remote-python --catalog --output`，以及 `--limit --mode-id --source-split --after-uuid --resume`。**实际执行会连接 SSH 和写本地数据，本文不提供现成账户、原盘路径或执行许可。** 不禁用 host-key 检查，不执行旧数据工程代码。

下游使用 [corpus_audit](../corpus_audit/README.md)重新审计；`CorpusFetchError`、路径/UUID 不一致或来源变化应停止，不能靠丢弃错误行伪装整批成功。

## 测试与状态

对应 [test_corpus_fetch.py](../../../tests/kar/test_corpus_fetch.py)。本轮未运行搬运、访问远端或读取真实 raw。
