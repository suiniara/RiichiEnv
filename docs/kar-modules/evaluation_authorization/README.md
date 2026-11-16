<!-- kar-module: riichienv.kar.evaluation_authorization -->
<!-- api: require_evaluation_authorization, read_evaluation_authorization, exclusive_job -->
# evaluation_authorization — 冻结作业许可与互斥租约

源码：[evaluation_authorization.py](../../../src/riichienv/kar/evaluation_authorization.py) · [总索引](../README.md)

## 主要接口

`read_evaluation_authorization(path)` 读取存在且大小受限的 JSON。`require_evaluation_authorization(document, operation, plan_sha256)` 验证 `explicit_frozen_evaluation_authorization_v1`、内容摘要、明确用户批准/引用、操作列表、精确计划与当前源码。

`exclusive_job(path)` 是上下文管理器，持有系统释放的 advisory file lock；Windows 与 POSIX 使用各自锁接口。进程异常结束时锁由系统释放，锁文件存在本身不表示当前仍被占用。

## 使用边界

标定、tau 验证、冻结导出、完整基准和 M3 需要各自的 exact operation 许可；写好代码/计划并不自动批准。plan/source 改变必须重新核验，不能只复用之前某个作业的 approval。

```python
from riichienv.kar.evaluation_authorization import require_evaluation_authorization

def check_frozen_job(approval_document, exact_plan_sha256):
    require_evaluation_authorization(
        approval_document, operation="run_m3", plan_sha256=exact_plan_sha256
    )
```

示例只检查许可，不能保证 plan 本身、价格、模型或数据已经合格，后续生产层还会验证这些条件。

## 错误与副作用

None、未知操作、未明确批准、来源/摘要不符抛 `PermissionError`。租约会创建/打开指定锁文件，应选择已批准输出/注册表目录，不放入 raw 原盘。该许可不是 OS 安全沙箱，也不是参数更新许可的替代。

## 测试与状态

对应 [production/test_jobs.py](../../../tests/kar/production/test_jobs.py)、[test_m3_registration.py](../../../tests/kar/production/test_m3_registration.py)。本轮未执行示例、获取作业锁或启动评测。
