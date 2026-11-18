<!-- kar-module: riichienv.kar.calibration_quality -->
<!-- api: BranchQualityCriterion, BranchQualityPolicy, CalibrationQualityPolicy, evaluate_branch_quality, evaluate_quality -->
# calibration_quality — 预登记总体与逐分支质量

源码：[calibration_quality.py](../../../src/riichienv/kar/calibration_quality.py) · [主设计 §11](../../KAR-v1.2.2-DESIGN.md) · [总索引](../README.md)

`CalibrationQualityPolicy` 保存原有 action/response/rank 总体 NLL/Brier/ECE 上限、最少观察数、门控保留率及来源，另显式包含 `branch_policy`。`evaluate_quality` 只检查总体，不用总体平均替代罕见分支。

`BranchQualityCriterion(head,event_id,max_nll,max_brier,max_ece,minimum_independent_games)` 针对 response 或 rank 的一个响应分支。`BranchQualityPolicy(criteria,source_reference)` 拒绝重复项，序列化指标总体、cal_check分区、独立游戏计数、缺失/回退合同、版本及内容摘要。`from_record` 严格校验字段/版本/摘要，不给旧记录补字段。

`evaluate_branch_quality(metrics,policy)` 为21个响应分支分别输出 response/rank 的 passed/failed/unavailable。指标使用真实 E 等于该分支的 cal_check 行：21类 response 或4类 rank NLL/Brier/10-bin top-label ECE；独立验证游戏按 game_id 去重。证据不足或未登记判据为 unavailable，充分证据但超阈值为 failed；两头均 passed 才通过该分支质量。

**没有正式默认阈值或验证样本数。** 必须在计算/检查 cal_check 前预登记；缺正式标准不冒称分支已校准。审查反例和测试数字只属合成 fixture，不能转成生产默认。

价值支持是 train≥500 与分支质量通过的交集，见 [support](../support/README.md)。失败/不可用不改变公开规则 mask、不删除或重新归一化响应概率，而是按既有−1.5和unsupported mass处理；无合格分支时 runtime 直接先验。总体失败仍执行既有整体回退。全部指标都是经验准入，不是 uniform error certificate。

实际接线与反例回归见 [test_branch_quality.py](../../../tests/kar/core/test_branch_quality.py)、[当前报告](../../kar-semantic-branch-repair.md)。本轮未做真实校准，不产生真实数据的参数更新或评测授权。
