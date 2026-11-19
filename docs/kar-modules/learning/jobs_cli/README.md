<!-- kar-module: riichienv.kar.learning.jobs_cli -->
<!-- api: main -->
# learning.jobs_cli — 冻结模型作业接线

源码：[jobs_cli.py](../../../../src/riichienv/kar/learning/jobs_cli.py) · [learning 索引](../README.md)

## 入口与职责

顶层 KAR 命令把 `m3 / benchmark / calibrate / select-tau / release` 转发到本模块。`main()` 读取 `--config` 和独立 `--authorization`，先核验 exact operation、plan SHA 和当前 source，再进入对应作业。

配置/不可变 artifact 受大小和存在性限制。不因提供了 JSON 就自动批准；此 CLI 不包含训练器或用户审批文件生成器。

## 配置分派

| job | 主要关联输入 |
|---|---|
| calibrate | checkpoint、dataset、validation_manifest、预登记 quality_policy、预算、温度网格/microbatch |
| select-tau | calibrated subject、三个对手、seed/grid/selection_policy、预算 |
| release | checkpoint、可选 calibration_job/tau_selection_job、policy 与实际 device |
| benchmark | subject/opponents、总 matches/seed/mode/cap、M1预算 |
| m3 | 完整 paired_seed_manifest、benchmark_report、固定对手/registry/output、M3预算与统计 |

通用字段含 `plan_sha256`、`output`，设备缺省 CPU。M3 manifest 可由路径读取，但其摘要必须等于审批计划。发布先重算 export plan 后加载，有限策略由共享 validator 再检查。

`calibrate` 和 `select-tau` 在任何 checkpoint/model/device 加载前解析并核验预算，向执行入口传递延迟工厂。入口先创建不可覆盖的启动身份、账本及未完成 attempt，再调用工厂。标定的数据全库 hash/索引和 tau 作业的四个发布加载均计入同一计算作用域；加载失败也记费，不另开内层账本重复计费。

## 副作用与失败

这些作业可加载网络、拟合温度、运行完整对局、写权重/数据/账本。标定/对局 no_grad 不等于没有计算，不能在“只写代码”阶段运行。缺 prerequisite、不同来源、质量失败或预算问题不会被自动降级成另一未登记策略。

生成计划使用 [planning](../../planning/README.md)，不要为了取得计划 hash 先执行实际 benchmark/M3。

## 测试与状态

对应 [production/test_jobs.py](../../../../tests/kar/production/test_jobs.py)、[test_review_job_guards.py](../../../../tests/kar/production/test_review_job_guards.py)、[test_frozen_evaluation_startup.py](../../../../tests/kar/production/test_frozen_evaluation_startup.py)、[test_metadata_planning.py](../../../../tests/kar/production/test_metadata_planning.py)。加载失败和硬退出回归只使用合成配置/授权、假加载器和零价格子进程，不启动正式作业。
