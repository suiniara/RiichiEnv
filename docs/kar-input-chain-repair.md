# KAR 输入链路：完整 LEGAL_SET、私有 SELF_PASS 与规范语义时钟

活动规范见[单一主设计](KAR-v1.2.2-DESIGN.md)，本轮结果见[修复与验收报告](kar-semantic-branch-repair.md)。本页说明实现，不是另一份可与主文档冲突的设计。原输入链轮a869→91b0db0的372项是历史证据。

## LEGAL_SET

每个有效kind8行增加 `mean(E_id+E_type+E_tile+E_tsumogiri+E_riichi)`。复用ACTION五表，无新可训练参数；`schema.action_factor_ids`共用因子工厂，只有DISCARD填后三因子，其他动作N/A。候选为完整全局ID，唯一非空≤32，按ID排序，过滤546，不Top-K、不合并变体。numeric16继续数量/32。

dense与encode_prefix共用embedding/_hidden；单ACTION后缀不再加整个集合。缓存绑定完整候选、参数/buffer版本、设备、上下文及当前input/model合同。Policy、Response、Rank三头在线性投影本身禁用autocast，保留9a4d154的FP32修复。

## 公共域和语义时钟

`event_id/cutoff_event_id`是公共流下标，不等于含私有marker的复合events数组下标。source_record_index/hash是来源证明，**不再分组推进模型时间**。

`event_semantics.PublicEventClock`由离线和在线账本共用：start_game、start_kyoku、首次摸牌分别占1/2/3；reach与discard、reach_accepted与后续draw、Dora各有明确语义tick。单事件展开多Token共位置。同触发多Ron共享裁定时间，公开支付只编码一次，在线不完整多Ron组拒绝决策快照；分块不能提前支付或丢掉后续赢家。

kan_commit是可选transport ACK，不占模型时钟；它可立即更新当前账本的已提交状态，但历史接受Token统一在共同可见的本次有效Dora/替换draw证据处生成。旧杠延迟Dora仍不证明新杠提交，有无ACK不改变最终历史。S/DECIDE位置=C+1，ACTION=C+2；H历史行数不再被当成时间C。`PublicSnapshot.public_clock_version`缺失/旧值拒绝，不为旧快照补当前版本。

## 私有Pass的因果边界

capture只捕获上下文；标准在线包装器在原子提交成功后确认。离线只有已验证非平凡选择与实际唯一正常下家draw，才推断Pass；不从user_input、未知意图或被覆盖鸣牌补造。

确认marker仅入本家，锚定实际公共关闭事件，不增加公共ID/时钟。杠的私有确认规范锚为替换draw，不被transport ACK先后左右。批次含后续事件时按锚插入，旧snapshot/TokenSequence不回写，其他观察者字节不变。当前决策可有后续标签，但该次输入不能含未来确认。

provenance保留native_committed_pass/inferred_unique_next_draw_pass；不同来源不必伪造相同provenance。相同可见事实、已知私有事实、候选与语义cutoff，全部模型字段（含positions）必须相同。

## 终局公开信息

九种九牌离线与在线共同验证宣布者、当前首次draw、14张幺九字条件、未舍牌未鸣牌、source tiles一致性。只在公开终止后揭示宣布者；不机械补出全知记录里的其他手牌。源tiles缺失时仅允许冻结规则与已验证前序事件的唯一重建，记录来源。已证明途中流局零转移与缺失日志分开，已知scores+零deltas产生四个present=true的SCORE_DELTA；缺失NoTile/Hule结算不填零。

必须比较到下一局的真实决策前缀，不能只比reason。历史缓存共享语义位置、支持增量与必要重建；不完整终局可见性不冒充可训练前缀。

## 当前版本与兼容性

宿主Schema/hash不变；artifact=`kar_riichienv_contract_v4`，input=`semantic_public_clock_terminal_reveal_v2`，model=`kar_transformer_v3_semantic_inputs`，train config=`explicit_train_split_scale_v4_semantics`。mask v3保留雀魂修正并补途中流局/末巡互斥；branch quality合同独立绑定。

旧v3输入的参数形状即使相同，位置/终局历史仍不同。旧数据、历史/KV缓存、checkpoint、校准、支持及发布不能精确复用，需要在新目录真实重建并另行授权；不手改旧hash或覆盖旧产物。未执行的真实数据、梯度、GPU、校准与棋力验收如实见当前报告，不沿用历史372/1348/45项宣布通过。
