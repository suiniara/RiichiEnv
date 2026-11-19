<!-- kar-module: riichienv.kar.learning.export -->
<!-- api: learning_contract, GameWriter, export_dataset -->
# learning.export — 从 M0 来源重建训练数据

源码：[export.py](../../../../src/riichienv/kar/learning/export.py) · [learning 索引](../README.md)

## 职责与接口

`export_dataset(acceptance_dir, proto_path, output_dir, ...)` 消费通过 gate 的 split 清单，重新解码/重放真实 raw，通过输入回调写共享历史决策数据。`GameWriter` 接收 `ReplayDecision`，另行构造 action/response/final-rank 监督；`learning_contract()` 给出派生数据必须匹配的版本。

实际加载的协议必须同时匹配验收 scope 与各局清单中的顶层 source SHA 和完整 imported descriptor SHA。SQLite scope、每局恢复元数据和输出 manifest 继续保存这组身份；恢复前严格核对，训练读取端也拒绝缺少任一摘要的 dataset。旧版只保存顶层协议 SHA 的产物需要重新审计、导出，不能仅更新合同版本或重新计算自签摘要。

这不是把旧学习数据库改个文件名，不会继承旧目录的 train/test 标记或训练资格。

## 数据合同

Golden/语义别名排除、主 cohort、game split、protocol/raw/receipt/acceptance 身份在导出前检查。舍牌的辅助响应来自完整实际窗口，value 来自完整半庄；缺目标保留有效位。可证 Pass 只监督动作头，诊断分别记录冻结窗口与标签证据 event ID；窗口触发可早于实际选择 cutoff（例如上一杠 Dora 到达后开放抢杠）。每局报告及 `response_opportunities_by_split` 记录非平凡机会、显式、唯一推断、删失与阶段／关闭切片，不把删失补成行动行。

导出 scope 使用 [versioning](../../versioning/README.md) 的统一合同，保留完整可用 split 数。manifest 分开保存切分前 `N_eligible_half_games`、实际导出的 `N_train_half_games`、`available_split_counts` 和 `exported_split_counts`，不得把探针子集冒充完整 cohort。主力门槛固定 `model_tier_basis=N_train_half_games`，实际 train rho 在完成导出后固定。完整清单接口不受诊断小批上限限制。角色 salt、杠事务、特征语义、随机置空、训练计划与损失版本均显式记录。

## CLI 与产物

顶层 `export-data` 分发到本模块。关键输入是 `--acceptance --proto --output`；其他选择参数以 `main()` 为准。输出为 `decisions.sqlite3` 与最后完整发布的 `manifest.json`，含共享 histories、games、samples 等表和数据库 SHA。

输出须在独立目录；原盘保持只读。导出会实际执行原生 replay/编码并写数据，不是纯元数据规划；未获本轮许可时不执行。

## 失败、测试与状态

选择被改、raw/协议不符、任一游戏重放失败、标签/mask 不一致或完整性缺失会阻断，不能丢坏行后报告完整成功。对应 [m1/test_dataset.py](../../../../tests/kar/m1/test_dataset.py)、[test_acceptance_shards.py](../../../../tests/kar/test_acceptance_shards.py)。本轮未对真实数据重新导出；合成来源/元数据及临时 SQLite 的合同回归结果见[实现覆盖报告](../../../kar-implementation-coverage.md)。
