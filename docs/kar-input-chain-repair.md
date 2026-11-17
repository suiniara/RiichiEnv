# KAR 输入链路：LEGAL_SET 与观察者私有 SELF_PASS

审查基线：`a8696698af57dfeeb4c2699c4b685eea56e19f77`，分支 `kar/riichienv-v1-20261117`。
主设计 SHA256：`4349039c44175ece77f6dddad3a2106fbd3e0e0339935122df242175c83cbdba`。

## 1. LEGAL_SET 的实际模型输入

旧前缀仅以 `numeric[16]` 表示候选数；模型没有读取 `ModelBatch.candidates`，等长集合无法影响输入。现在每个有效 LEGAL_SET(kind=8) 行增加：

`mean_a(E_id[a] + E_type[type(a)] + E_tile[tile(a)] + E_tsumogiri[tsumogiri(a)] + E_riichi[riichi(a)])`。

- 复用 ACTION 的五张既有表，没有新的 Embedding/可训练参数。
- `schema.action_factor_ids()` 与 ACTION 后缀共同派生五个索引。只对 DISCARD 填牌种/摸切/立直因子；所有非舍牌按冻结 §4.7 使用 N/A，不把 `a//4` 应用于全 546 类。
- 模型输入过滤 padding=546，真实成员必须唯一、非空、容量不超过 32；重复真实 ID 拒绝，而不是重复加权。按 ID 排序后求和，集合排列不改变表示。
- 保留 `numeric[16]=|C|/32` 的原数量语义，不截断、不只池化舍牌、不合并四种变体。
- dense 与 `encode_prefix` 都经过 `_hidden` / 同一 InputEmbedding。单 ACTION 的 `score_actions` 不再注入整个集合；它只验证集合与缓存所属前缀相同。
- K/V cache 新绑定完整候选集合、输入版本及模型合同；参数或计算 buffer 被改动、旧输入版本或上下文不一致均拒绝。

固定可控 Embedding 的回归直接读取实际 embedding 输出，精确核对公式；不是只检查输出 legal mask。

## 2. 公共事件域、私有历史与逻辑时钟

`ObserverLedger.event_id` / `PublicSnapshot.cutoff_event_id` 继续表示规范化**公共事件流**下标；它不是复合 `snapshot.events` 的数组下标。`source_record_index` 只是公共可见逻辑组，不等于 event ID。

观察者自己的 `events`/`history_positions` 可以多出已确认私有标记。SELF_PASS：

- 不增加公共 event ID，不推进公共逻辑时钟，不修改手牌、河牌、副露、分数或他家状态。
- 放在关闭相应响应窗口的公共事件之后，使用该确认事件的逻辑位置。同组多个 Token 共位置本来就受完整 Schema 支持。
- 因 H 增加一行，历史行数改变；S/DECIDE 的逻辑位置仍由公共可见时钟 C 决定，不因别人的私有事件变化。
- 先前 `PublicSnapshot` 与 `TokenSequence` 是冻结副本；确认后的 materialization 不改写它们。
- 若一个原生 batch 已包含后续 Dora、摸牌或新局，按公共确认锚插入本家的复合历史，不把旧局 Pass 放到新局后。增量缓存能延伸新尾部；前缀不再相符时重建，不输出陈旧 blob。

## 3. 确认入口与在线事务

`SelfPassTrigger` 是观察者专属的已知触发上下文，不是成功凭证。`pass_trigger()` 只捕获当前普通舍牌或已声明杠的 source/kind/pai/public ID；加杠要求原 PON 存在，暗杠牌信息使用该观察者已过滤的公开表示。

`record_self_pass(trigger, provenance=...)` 仅接受本家/已观察/已关闭窗口及支持的来源；`apply_event(type=self_pass)` 有同样的 owner/source/confirmed/时间校验，并在任何公共状态推进前验证。来源为：

- `native_committed_pass`：执行包装器已验证合法选择且原子原生提交成功。
- `inferred_unique_next_draw_pass`：冻结的离线唯一正常摸牌响应规则，不是隐藏意图猜测。

触发 public ID 是本家去重键；重复确认 no-op，既不重复插入，也不改写首次 provenance。

`run_match` 在响应提交前调用 `capture_self_passes`（无副作用），只有 `execute_joint_responses` 正常成功后才向 `observe_native` 提供这些已确认选择。失败、取消不确认。Projection 先完成公共批次校验，再按确认锚只写各自账本；公开 `projection.events` 不包含 self_pass，原生 public stream 中混入私有事件立即拒绝。

该入口是可信调用方的记录合同，不是数字认证；捕获 trigger 或设置字符串本身都不证明提交成功。标准执行链保证成功边界，不从他家动作猜本家行为。

## 4. 离线监督与历史分离

原有 `inferred_unique_next_draw_pass` 监督规则保留：正常 NEXT_DRAW 关闭窗口、该观察者确有非平凡 Pass 选择，才能产生该标签。优先级覆盖下未成立的 CHI/PON/RON 不一律变成 Pass，raw user_input 不用于补行为。

Pass 决策样本仍冻结于触发舍牌后、ACK/摸牌前。可用后续事实作监督标签，但私有历史仅在循环真正处理到确认摸牌时写入。于是：

- 当前 Pass 前缀不含本次 SELF_PASS；
- 下一次本家输入恰有一条该窗口的 kind=26；
- 较早 cutoff 不会被未来 NEXT_DRAW 增补；
- 有无 export callback 都得到相同后续输入，provenance 保留在标记/样本元数据中，不变成 E/R 模型输入。

## 5. 输入版本与兼容性

完整 Schema ID、canonical hash、546 动作、21 响应、三头、既定 loss、雀魂 mask v2 均不改：

- `schema_id=kar-riichienv-v1`
- canonical hash `fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673`
- `response_mask_version=public_rule_response_mask_v2_mjsoul_nagashi`

但 forward 与派生历史改变，因此活动身份更新为：

- `artifact_contract_version=kar_riichienv_contract_v3`
- `input_pipeline_version=legal_set_pool_private_self_pass_v1`
- `MODEL_VERSION=kar_transformer_v2_legal_set`
- `training_config_version=explicit_train_split_scale_v3_inputs`（recipe 包含 input 版本）

旧 v1/v2 权重产物即使参数形状一致也不能精确续训/评估；标准 checkpoint、数据、支持/校准、发布加载合同拒绝缺失/不匹配字段，旧成本 recipe 不能静默复用。旧 K/V 和历史缓存需重建。Token 编码在没有新增已知 Pass 的同一快照上可保持字节不变，但这不证明整条旧派生数据/旧 forward 精确兼容。

本轮不修改旧 manifest/hash、不迁移权重、不重导出数据、不做真实校准。手工绕过加载器或给旧产物换标签不构成兼容迁移。

## 6. 定向证据

本轮 372 个有界 CPU forward/历史/版本回归通过，0 failed/errors/skipped。所有最终组均记录执行前后相同的源码/测试文件指纹、完整 argv、退出码、原始 stdout、JUnit 和硬禁止调用计数。上一轮 1348 项是历史证据，不替本次结果。

实际包含精确 LEGAL_SET 向量、等长差异、置换/padding/非舍牌、dense/cache 与 teacher 隔离、全部 preset 参数量、普通及抢暗杠真实 Pass、受控 run_match、失败/取消、争议 CHI、KAKAN 原 PON、批量/增量与离线对拍。

未启动完整对局 runner、阶段任务或 M3；新增 Pass 回归均为有界合成情景。既有 `test_token_cache` 另读取一个已存在的固定牌谱资产，只做只读 Token 增量/批量对拍，不导出数据、不把它当作新增私有 Pass 的真实证书。

未执行 backward、autograd.grad、optimizer.step、M1/M2/M3、训练 launcher、GPU/云、真实校准或数据重导出。守卫记录上述禁止调用为零。未执行梯度验证意味着本轮不声称新增池化的训练梯度已验收；只确认 forward 共用实现。至此停止。
