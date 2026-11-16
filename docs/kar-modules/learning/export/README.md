<!-- kar-module: riichienv.kar.learning.export -->
<!-- api: learning_contract, GameWriter, export_dataset -->
# learning.export — 从 M0 来源重建训练数据

源码：[export.py](../../../../src/riichienv/kar/learning/export.py) · [learning 索引](../README.md)

## 职责与接口

`export_dataset(acceptance_dir, proto_path, output_dir, ...)` 消费通过 gate 的 split 清单，重新解码/重放真实 raw，通过输入回调写共享历史决策数据。`GameWriter` 接收 `ReplayDecision`，另行构造 action/response/final-rank 监督；`learning_contract()` 给出派生数据必须匹配的版本。

这不是把旧学习数据库改个文件名，不会继承旧目录的 train/test 标记或训练资格。

## 数据合同

Golden/语义别名排除、主 cohort、game split、protocol/raw/receipt/acceptance 身份在导出前检查。响应来自完整实际窗口，value 来自完整半庄；缺目标保留有效位，不能伪造 NEXT_DRAW 或按结果修改输入。

导出 scope 保留可用 split 数与是否 probe subset；实际 train rho 在完成导出后固定。完整清单接口不受诊断小批上限限制。角色 salt、杠事务和输入/损失版本都属于当前合同。

## CLI 与产物

顶层 `export-data` 分发到本模块。关键输入是 `--acceptance --proto --output`；其他选择参数以 `main()` 为准。输出为 `decisions.sqlite3` 与最后完整发布的 `manifest.json`，含共享 histories、games、samples 等表和数据库 SHA。

输出须在独立目录；原盘保持只读。导出会实际执行原生 replay/编码并写数据，不是纯元数据规划；未获本轮许可时不执行。

## 失败、测试与状态

选择被改、raw/协议不符、任一游戏重放失败、标签/mask 不一致或完整性缺失会阻断，不能丢坏行后报告完整成功。对应 [m1/test_dataset.py](../../../../tests/kar/m1/test_dataset.py)、[test_acceptance_shards.py](../../../../tests/kar/test_acceptance_shards.py)。本轮未导出或重放数据。
