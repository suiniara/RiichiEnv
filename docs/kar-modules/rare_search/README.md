<!-- kar-module: riichienv.kar.rare_search -->
<!-- api: RareSearchError, SearchConfig, freeze_field_map, search -->
# rare_search — 稀有规则场景候选定位

源码：[rare_search.py](../../../src/riichienv/kar/rare_search.py) · [总索引](../README.md)

## 职责与接口

在有限数量记录中定位可能覆盖稀有规则的来源，不在远端执行完整训练/Golden 验收。

- `SearchConfig`：连接信息、catalog、proto、最多扫描数、旧 split/UUID 游标。
- `freeze_field_map(protocol)`：根据本地核验协议固定远端只读解析所需字段映射及 hash。
- `search(config, output_dir, resume=False)`：执行有界候选搜索并持久化位置/错误/进度。
- `RareSearchError`：搜索合同、恢复或来源失败。

## 关键合同

`max_scanned` 仅为 1..10000；`SearchConfig.scope()` 将选择角色固定为 `golden_only`。命中候选不表示规则已核验，不能标成真实 Golden PASS，也不能把候选游戏加入训练。

远端脚本只定位必要结果字段，沿用 [corpus_fetch](../corpus_fetch/README.md)的路径、SSH 和只读连接约束。`.user_input` 不参与定位或标签。proto/field map/远端脚本 hash、来源范围和游标变化时不能直接续用旧 checkpoint。

## CLI 与产物

通过 `python -m riichienv.kar.rare_search` 使用 `--host --ssh-executable --remote-python --catalog --proto --output`；可指定 `--max-scanned --mode-id --source-split --after-uuid --resume`。执行会真实连接指定数据机，不能因 README 示例而自动运行。

输出为候选和错误日志、进度及来源信息。后续需要下载相应 raw，再用 `corpus_audit` / `golden` 对完整结果、独立结算与冻结期望重验；不能用此模块替代验收。

## 测试与状态

对应 [test_rare_search.py](../../../tests/kar/test_rare_search.py)。本轮未扫描、未连接远端；缺少真实案例仍是证据缺失，不能改为 N/A 或生成 synthetic 替代。
