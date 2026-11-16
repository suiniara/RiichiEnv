<!-- kar-module: riichienv.kar.learning.prerequisites -->
<!-- api: m2_prerequisites, verify_prerequisites -->
# learning.prerequisites — 重验真实 M1→M2 证据

源码：[prerequisites.py](../../../../src/riichienv/kar/learning/prerequisites.py) · [learning 索引](../README.md)

## 职责与接口

`m2_prerequisites(m1_report, config, dataset_sha256, tau_selection_job=None)` 从实际文件建立 M2 前置证据；`verify_prerequisites(record, config, dataset_sha256)` 重建并精确比较已登记来源，防止登记后换数据、报告或权重。

模块不创建网络/优化器，不导入 Torch 进行模型运行；会读取产物、检查点字节、native/source 身份和共享账本，不是只看一个 PASS 字段。

## 检查链

- M1 summary 内容 hash、phase/完成标识、同初始化/顺序/更新数声明与两份实际 run_manifest 一致。
- feature-on/off 均 Small、各一完整 Epoch，实际 train/eligible 数、数据 manifest 和模式匹配。
- 完整 validation、当前 source/native 构建、generation 路径及 payload SHA、原共享 M1 费用锚点均有效。
- Medium/Large 扩规模额外要求生产 tau v3 job、cal_fit 而非 test/cal_check、预登记收益通过，且 binding 对应实际 M1 checkpoint 与相同特征模式。

## 输出与拒绝条件

返回带源路径、报告/运行摘要、验证 job 身份、数据/源码/native、目标 tier/模式及 `growth_validated` 的完整 record。固定合同为 `completed_m1_to_m2_gate_v1`。

缺证据时拒绝执行，而不是伪造或自动重跑 M1。Small 的 M2 仍需要完整 M1，但“没有验证收益不得扩模型”不会被错误解释为已经获得 Medium/Large 资格。

## 接线、测试与状态

由 [orchestration](../orchestration/README.md)生成，经 [training_config](../training_config/README.md)登记和 raw trainer 再核验。对应 [production/test_stage_prerequisites.py](../../../../tests/kar/production/test_stage_prerequisites.py)、[test_review_finance_and_resume.py](../../../../tests/kar/production/test_review_finance_and_resume.py)。本轮未读取真实阶段产物或签发可执行证据。
