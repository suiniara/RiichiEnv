<!-- kar-module: riichienv.kar.budget -->
<!-- api: CostForecast, FixedCostReceipt, BudgetPlan, BudgetLedger, BudgetExceededError, choose_training_tier -->
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

## 记账语义

M1/M2/M3 分别不超过 USD100/650/250，总额1000。`spent_usd` 是已记账费用；计算费由实测 elapsed 与登记单价结算，不表示已自动核对云供应商账单。`reserved_fixed_usd` 是尚未真实对账的固定成本余量；`committed_usd=spent+reserved` 才用于后续放行。

固定费预留持久保存。`charge(..., receipt_id=...)` 支持唯一收据幂等对账；重复 ID 内容不同拒绝。实际超支也先保留已发生收费，再抛错，不能隐藏。

## 执行与恢复

`reserve_next` / `assert_room` 在工作前检查余量；`settle_elapsed` 在边界结算；`record_compute(reference)` 成功及可捕获异常均结算最后计算，不把失败部分免费丢弃。`state_dict`、`verify_anchor`、`verify_state` 防止保留完成收据却删掉/回退账本。

账本 v2 不静默迁移旧格式。费用引用、实际单价和其他项目支出由操作者提供并负责核验；此模块不是外部云账单服务或安全沙箱。M2/M3 的唯一注册还负责固定具体账本路径。

## 测试与状态

对应 [production/test_planning.py](../../../tests/kar/production/test_planning.py)、[test_review_finance_and_resume.py](../../../tests/kar/production/test_review_finance_and_resume.py)、[test_review_job_guards.py](../../../tests/kar/production/test_review_job_guards.py)。本轮未记账、消费预算或运行测试。
