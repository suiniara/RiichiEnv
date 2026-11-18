# KAR / RiichiEnv 设计一致性与迁移合同

**唯一活动主设计：** [KAR-v1.2.2 / RiichiEnv V4](KAR-v1.2.2-DESIGN.md)。[迁移说明](kar-design-v4-migration.md)和[文档校验](kar-design-v4-validation.md)记录 `f70684e` 对应的 `design-sync-1`，不是现行版本表。后续振听与校准兼容边界见[最新修复报告](kar-furiten-event-calibration-repair.md)。

## 1. 完整宿主 Schema 与来源

活动 Schema 保持 `kar-riichienv-v1`，完整 canonical SHA=`fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673`。设计引用仍为 `kar-v1.2.2`，overlay真实字节SHA=`90286d088014eb72e9f99f1b3c35d0a7553bcf5fec99e8aeb3ba5f02a010716a`；它不是完整Schema哈希，也不是可接受的活动ID别名。

映射 `riichienv_complete_v1_from_kar_v1_2_2_v1`、legacy兼容=`not_claimed`。139字节/944行/62槽、动作/响应身份、模型形状及三项损失不变。不存在要求实施者继续寻找缺失Kanachan父registry的活动阻断。

原附件SHA=`4349039c44175ece77f6dddad3a2106fbd3e0e0339935122df242175c83cbdba`。原件保留为历史来源；当前实施以仓库内完整修订主文件为准，而不是外部目录中的未修订文本。

## 2. 模型规模冲突已经在主文档正文解决

原附件§7.5/14.2按切分前N_eligible，§17按train，二者曾矛盾。用户已明确选择保守的 **N_train_half_games**；本轮主文档所有规模表、预算/launcher字段和发布顺序统一按该口径，不再留给实施者自行选择。

N_eligible是审计后切分前完整池；available_split_counts是池的整场split；exported_split_counts是当前实际导出量；N_train是当前实际导出的train完整半庄。<30000 Small，30000..49999 Medium，≥50000 Large；实测预算不足依次降档或停止。正式M2不能用probe子集冒充全量train/validation。

## 3. V4 的静态合同

完整活动版本与源定义仅见[主设计](KAR-v1.2.2-DESIGN.md) §0；迁移依赖见 §19。不得补字段、重签旧 hash 或沿用旧通过数来冒充迁移。旧报告 [V4 输入与分支质量](kar-semantic-branch-repair.md)及[分批 Pass / 有效未支持质量](kar-pass-rho-math-repair.md)各自记录其原验收范围。

## 4. 当前证据边界

[历史文档校验](kar-design-v4-validation.md)只验证当时正文、源码对应关系和分发快照，不重新验证旧运行报告。新修订的实际运行与未执行项见[本轮修复报告](kar-furiten-event-calibration-repair.md)。过去执行过的更新和梯度诊断仍以其历史报告为准。
