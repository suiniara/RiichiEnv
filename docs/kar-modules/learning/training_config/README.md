<!-- kar-module: riichienv.kar.learning.training_config -->
<!-- api: ProductionTrainConfig, training_recipe_sha256, main_model_registration, registered_main_model_ledger, validate_main_model_registration, third_epoch_allowed -->
# learning.training_config — 正式配方、登记和第三轮条件

源码：[training_config.py](../../../../src/riichienv/kar/learning/training_config.py) · [learning 索引](../README.md)

## 主要接口

`ProductionTrainConfig` 记录 phase、实际 N_train/N_eligible、tier、planned_epochs、预算、seed、设备/精度、microbatch、特征模式和固定超参。M1 仅 Small 一轮，M2 为两轮/条件第三轮；有效 batch Small/Medium/Large 为128/256/512。

`training_recipe_sha256(arguments)` 先于价格生成标准化配方身份；`to_record/from_record` 校验往返派生字段。训练计划仍为 `budget_ladder_v1`，配置序列化另用 `explicit_train_split_scale_v4_semantics`，必须显式包含新 `input_pipeline_version`、`model_tier_basis=N_train_half_games`、规则置空版本及概率；不再以默认值补齐缺失的旧版本字段。两个 N 字段禁止 bool/负数混入，M2 不得因 eligible 总量较大就越过 train 门槛。

## 唯一登记

`main_model_registration(dataset_sha, config, output_root, prerequisites=None)` 生成 v3 登记，固定实际 training 输出、M2 共享账本、配方和前置证据。没有 prerequisites 的草稿 `execution_ready=false`。

`validate_main_model_registration(...)` 校验全部身份/路径，并重读真实 M1/验证来源。原始 trainer 也执行它，因此不能绕开上层调度器换目录重置预算或跳过 M1。

`registered_main_model_ledger(...)` 仅从已签名登记校验数据/配方、唯一输出及账本路径和非草稿状态，供 trainer 在完整前置验证之前绑定原费用账本；它不代表前置证据已经核验。trainer 持久登记 compute attempt 后仍必须调用完整 `validate_main_model_registration(...)`，因此重读检查点的费用归入该 attempt，任何验证失败都会在模型分配前终止。

## 第三轮 gate

`third_epoch_allowed(previous, current, tolerance, budget_has_room=...)` 要求完整 validation 报告，但**只使用其中 model_selection/cal_fit 子记录**。角色 salt、cal_check_used=false、同一游戏集合、完整 cal_fit、总目标改善、各头登记恶化容忍值及预算都须满足。

缺指标不当零；cal_check/test 或不同游戏集合不能决定追加一轮。它是已登记的条件分支，不是重新搜索超参。

## 依赖、测试与状态

配置本身不创建网络；运行前置核验可能读检查点/账本元数据。对应 [production/test_planning.py](../../../../tests/kar/production/test_planning.py)、[test_stage_prerequisites.py](../../../../tests/kar/production/test_stage_prerequisites.py)、[test_review_finance_and_resume.py](../../../../tests/kar/production/test_review_finance_and_resume.py)。本轮只有临时合成登记/产物的单位回归，没有真实训练登记或阶段运行。
