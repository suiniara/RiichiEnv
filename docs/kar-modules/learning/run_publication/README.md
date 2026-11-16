<!-- kar-module: riichienv.kar.learning.run_publication -->
<!-- api: performance_summary, completed_run_record, publish_run_manifest -->
# learning.run_publication — 完成结果的可恢复发布

源码：[run_publication.py](../../../../src/riichienv/kar/learning/run_publication.py) · [learning 索引](../README.md)

## 接口与职责

`completed_run_record(state, ...)` 仅对已有验证历史、完成 Epoch 且 `finished=true` 的权威状态生成结果；核对 step、release contract、最终权重 hash，以及存在时的性能计数/训练进度一致性。

`publish_run_manifest(path, record)` 补齐内容摘要，以临时文件、flush/fsync 和原子 replace 发布。已有不同或损坏内容先按字节 hash 留存历史，不静默抹掉证据。

`performance_summary(value)` 以累计真实样本/tokens除以累计已提交更新耗时，输出吞吐和真实 microbatch padding 比例。

## 数字含义

不能对不同大小/时间的每步 samples/s 做算术平均。dummy 不进真实 tokens/样本分子，但其计算开销仍在 wall-time 分母。训练耗时、数据等待总量和验证耗时分开保存。缺历史性能记录返回 None，不伪造测量数字。

## 恢复场景

最后检查点成功提交后进程中断，JSON 完成发布尚未发生：trainer 加载权威 generation 后重新验证并调用本模块，标记 `recovered_completed_checkpoint=true`，**不重做 forward/backward/step**。

本模块是纯元数据与文件发布层，不导入 Torch；它不证明权重训练正确，也不直接读取模型 Tensor。实际权重 hash 和检查点证据由调用方提供。

## 错误、测试与状态

未完成状态、计数/step/最终权重矛盾、冲突历史归档或发布失败会拒绝。对应 [production/test_review_finance_and_resume.py](../../../../tests/kar/production/test_review_finance_and_resume.py)、[test_full_trainer.py](../../../../tests/kar/production/test_full_trainer.py)、[test_diagnostic_distributions.py](../../../../tests/kar/production/test_diagnostic_distributions.py)。本轮未写模型运行结果或执行测试。
