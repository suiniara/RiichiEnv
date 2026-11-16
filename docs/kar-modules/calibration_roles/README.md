<!-- kar-module: riichienv.kar.calibration_roles -->
<!-- api: calibration_is_fit -->
# calibration_roles — 不变的 game 级验证角色

源码：[calibration_roles.py](../../../src/riichienv/kar/calibration_roles.py) · [总索引](../README.md)

## 职责与接口

`calibration_is_fit(game_id, salt=CALIBRATION_SPLIT_SALT) -> bool` 依据 `salt + NUL + game_id` 的 SHA-256 前八字节奇偶，确定该验证游戏属于 cal_fit 还是 cal_check。默认 salt 为 `kar-cal-fit-check-v1`。

合同版本 `immutable_cal_fit_selection_cal_check_diagnostics_v1`。模块只分配角色，不判断一个游戏是否属于主 validation；主 split/cohort/Golden 排除仍由数据与校准 manifest 校验。

## 使用与边界

```python
from riichienv.kar.calibration_roles import calibration_is_fit

def role_of(verified_validation_game_id):
    return "cal_fit" if calibration_is_fit(verified_validation_game_id) else "cal_check"
```

同一游戏的所有决策/观察者角色一致，重复行不产生新样本角色。输入不可为空。低层函数保留显式 salt 参数供受控诊断，但生产校准拒绝换 salt；不能在训练或查看 cal_check 后重新 hash，以把不利样本搬进另一分区。

## 接线

[calibration](../calibration/README.md)用于温度/门控与产物 split；[evaluation](../learning/evaluation/README.md)只从 cal_fit 构造模型选择统计；[training_config](../learning/training_config/README.md)第三轮 gate 检查同一角色。全 validation 可以完整报告，但 cal_check 不参与选择。

无 Torch、随机抽签、文件写入或独立 CLI。对应 [production/test_selection_and_roles.py](../../../tests/kar/production/test_selection_and_roles.py)、[production/test_planning.py](../../../tests/kar/production/test_planning.py)。本轮未执行示例或测试。
