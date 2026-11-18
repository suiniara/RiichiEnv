# KAR / RiichiEnv 设计一致性与迁移合同

**唯一活动主设计：** [KAR-v1.2.2-DESIGN.md](KAR-v1.2.2-DESIGN.md)。已把宿主协议、公共时钟、N_train选型、输入/规则/质量版本及分支回退直接合入正文。此页是映射与来源说明，不是第二份主设计。

## 1. 完整宿主 Schema 与来源

活动 Schema 保持 `kar-riichienv-v1`，完整 canonical SHA=`fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673`。设计引用仍为 `kar-v1.2.2`，overlay真实字节SHA=`90286d088014eb72e9f99f1b3c35d0a7553bcf5fec99e8aeb3ba5f02a010716a`；它不是完整Schema哈希，也不是可接受的活动ID别名。

映射 `riichienv_complete_v1_from_kar_v1_2_2_v1`、legacy兼容=`not_claimed`。139字节/944行/62槽、动作/响应身份、模型形状及三项损失不变。不存在要求实施者继续寻找缺失Kanachan父registry的活动阻断。

原附件SHA=`4349039c44175ece77f6dddad3a2106fbd3e0e0339935122df242175c83cbdba`。原件保留为历史来源；当前实施以仓库内完整修订主文件为准，而不是外部目录中的未修订文本。

## 2. 模型规模冲突已经在主文档正文解决

原附件§7.5/14.2按切分前N_eligible，§17按train，二者曾矛盾。用户已明确选择保守的 **N_train_half_games**；本轮主文档所有规模表、预算/launcher字段和发布顺序统一按该口径，不再留给实施者自行选择。

N_eligible是审计后切分前完整池；available_split_counts是池的整场split；exported_split_counts是当前实际导出量；N_train是当前实际导出的train完整半庄。<30000 Small，30000..49999 Medium，≥50000 Large；实测预算不足依次降档或停止。正式M2不能用probe子集冒充全量train/validation。

## 3. 本次活动合同

artifact v4，input=`semantic_public_clock_terminal_reveal_v2`，公共时钟=`public_semantic_clock_v1`，终局公开=`verified_abort_public_resolution_v1`，mask=`public_rule_response_mask_v3_abort_last_draw`，branch quality=`preregistered_response_branch_quality_v1`。完整版本表以主文档§0及 `versioning.artifact_contract()` 为准。

保留 LEGAL_SET 五表均值、私有确认SELF_PASS、三头FP32线性投影、雀魂流局满贯及三态特征依赖修正。时钟只由规范语义事件决定，source record/native批次只作审计信息。mask只排除公开规则不可能；统计质量独立进入有效价值支持，不混入mask。

旧数据/缓存/checkpoint/支持/校准/发布缺任一新合同字段或版本不一致都拒绝；不能补字段、重签旧hash、沿用旧PASS冒充迁移。真实重建需求见[本轮报告](kar-semantic-branch-repair.md)，未自动执行。

## 4. 证据与历史

本轮结果见[当前修复与验收报告](kar-semantic-branch-repair.md)。既有a869一致性轮1348项、输入链轮372项、9a4缓存精度轮45项均是各自源码的历史证据，不是V4验收。过去实际执行过的更新/梯度诊断不改口为“从未训练”；本轮禁止反向/更新及阶段任务，按新守卫和日志单独报告。
