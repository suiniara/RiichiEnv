<!-- kar-module: riichienv.kar.budget -->
<!-- api: CostForecast, FixedCostReceipt, BudgetPlan, BudgetLedger, BudgetExceededError, PendingComputeError, choose_training_tier -->
# budget — 分阶段成本、持久预留与费用锚点

源码：[budget.py](../../../src/riichienv/kar/budget.py) · [总索引](../README.md)

## 主要接口

| 接口 | 职责 |
|---|---|
| `CostForecast` | 完整工作量、实测速率、登记单价、固定成本、测量引用、recipe 和 safety factor |
| `BudgetPlan` | phase cap、初始支出、其他阶段支出和真实固定费收据 |
| `BudgetLedger(path, plan, ...)` | hash-chain JSONL，恢复并保存预留与已记账费用 |
| `FixedCostReceipt` | 唯一 ID 的真实非计算收据，不能用预测伪装 |
| `choose_training_tier(...)` | 依实际 N_train 与可负担的已测预测保守选型 |
| `BudgetExceededError` | 不借预算、不悄悄减工作量的中止 |
| `PendingComputeError` | 旧计算 attempt 尚未核销，禁止用新进程时钟继续计算或伪装结清 |

## 记账语义

M1/M2/M3 分别不超过 USD100/650/250，总额1000。`spent_usd` 是已记账费用；正常计算费由实测 elapsed 与登记单价结算，不表示已自动核对云供应商账单。`reserved_fixed_usd` 是尚未真实对账的固定成本余量；`reserved_compute_usd` 是尚未结算的计算估算余额。`committed_usd=spent+reserved_fixed+reserved_compute` 才用于后续放行。估算不进入 `spent_usd`。

固定费预留持久保存。`charge(..., receipt_id=...)` 支持唯一收据幂等对账；重复 ID 内容不同拒绝。实际超支也先保留已发生收费，再抛错，不能隐藏。

## 执行与恢复

开始计算前调用 `reserve_next(seconds)`，先以 fsync 持久化一个唯一 compute attempt。没有可依据的初始化耗时预测时可用 `reserve_next(0.0)`：金额为零，但仍是未结束的计算，不能在硬中断后被当作已证明的零费用。`record_compute(reference)` 会自动开这个零金额标记，并在正常退出或可捕获异常时调用 `finish_compute`。

正式 trainer 的 attempt 从完整数据 manifest/SQLite 读取前开始，覆盖全文件 SHA、索引及长度物化、M2 注册前置验证、设备/进程组和模型初始化。数据身份先取自已核验授权，完整数据合同随后在已记账的区间内复核；失配同样保留已发生费用。预处理结算后仍保持 pending 状态，费用超限则停止后续 group/model 分配。多 rank 必须先获得绑定本次 launch 和全部 rank nonce 的 primary 收据，才能进入预处理。

M1/M2 调度器的设备/进程组 bootstrap 使用同一 phase 账本和既有 forecast/job 身份，交给 trainer 前关闭该短 attempt；内层再开新 attempt，固定费不重复预留，时钟区间也不重叠。调度器的 `.phase-startup/startup.json` 保存身份和账本锚点。此范围不追溯外部调用方已建立进程组的历史费用，也不覆盖调度器在启动阶段前已执行的证据预检。

同一活跃 attempt 多次 `reserve_next` 不累加整份预测：保留额更新为 `max(原余额, 当前未结算elapsed费用 + 下一段预计费用)`。已持有足够余额时无需追加记录。`fixed_usd` 仍只是额外的准入余量检查；需要持久保存的非计算费用使用 plan 的固定预留及真实固定收据。

`settle_elapsed(reference)` 用**一条日志事件**同时写入已测费用并扣减计算预留，保持 attempt 未结束。这样在两个结算边界之间被强制终止，仍会留下未结束标记。`finish_compute(reference)` 用一条事件结算最后一段并关闭当前实例拥有的 attempt；只有完成或可捕获的终止才能调用。已测实际超支先入账再抛错，时钟也随已提交事件推进，避免 abort 路径重复收费。

重开账本时，旧 attempt 不属于新实例。`pending_compute` / `state_dict()["pending_compute"]` 提供其 `attempt_id`、`job_id`、预留余额、登记费率、forecast 身份及已经结算的金额。即使旧预留余额为零，`assert_room`、`reserve_next`、`settle_elapsed`、`finish_compute` 都会拒绝继续计算；新 job 名称也不能绕过。读账本和核对 checkpoint anchor 仍可进行。

### 硬中断后的核销

先保留原账本、checkpoint 与外部实例运行记录，确认旧进程已经结束。使用原 job 的 plan 打开账本，读取旧 attempt，取得能核实**上次持久结算之后、尚未记入账本的实际费用**的外部凭据，再调用：

```python
ledger.reconcile_compute(
    attempt_id,
    usd=verified_unrecorded_cost_usd,
    receipt_id=external_receipt_id,
    reference=external_evidence_reference,
)
```

`usd` 不是原预测，也不是已计费部分重复在内的整份 job 账单；需要从外部总额中核对并扣除已记录费用。即使余款为零，也必须有依据。该操作原子地追加真实余款并释放旧预留，唯一 receipt 可幂等重放，内容改变则拒绝。不得通过普通 `settle_elapsed`、改 job 名、重置费用或删除账本来关闭旧 attempt。没有实际证据时保持阻塞，不能把预测直接写成实际收费。核销后原 checkpoint anchor 仍是同一追加历史的一部分。

### 写入中断与兼容性

每次追加有短暂 OS advisory lease 和已读文件长度核对，陈旧的 ledger 实例必须重开，不能覆盖另一个 job 的新增事件。若写入或 fsync 返回不确定错误，当前实例禁止重试计费；重开后按完整、校验有效的持久事件恢复。末尾半行、损坏的完整记录或错误 hash 链均明确拒绝并保留原文件，**不自动忽略或截掉尾部**；文件级修复需要保留原件及独立的审计证据。

账本当前为 `durable_compute_cost_ledger_v3`；v2 及旧 checkpoint 的账本状态不静默升级、不改写历史或重签 hash。没有提供自动迁移工具。费用引用、真实余款、登记单价和其他项目支出由操作者提供并负责核验；此模块不是外部云账单服务或安全沙箱。M2/M3 的唯一注册仍固定具体账本路径。

## 测试与状态

对应 [production/test_budget_compute_recovery.py](../../../tests/kar/production/test_budget_compute_recovery.py)、[test_planning.py](../../../tests/kar/production/test_planning.py)、[test_review_finance_and_resume.py](../../../tests/kar/production/test_review_finance_and_resume.py)、[test_review_job_guards.py](../../../tests/kar/production/test_review_job_guards.py)。新回归只用临时账本、合成时钟/价格和子进程硬退出，覆盖重复预留、部分结算、零金额标记、外部核销、普通异常、超支、写入结果不确定、坏尾部、陈旧 writer 和旧版本拒绝；不启动模型、参数更新、GPU 或真实付费任务。
