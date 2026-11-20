<!-- kar-module: riichienv.kar.learning.precision -->
<!-- api: FrozenBackendPrecision, capture_frozen_backend_precision, require_frozen_backend_precision, frozen_fp32 -->
# learning.precision — 冻结 IEEE FP32 后端合同

源码：[precision.py](../../../../src/riichienv/kar/learning/precision.py) · [learning 索引](../README.md)

## 元数据与运行时职责

`FrozenBackendPrecision` 是不可变的 `frozen_fp32_backend_v1` 元数据合同；`from_record()` 验证保存的版本、字段和 IEEE FP32 要求，`to_record()` 提供用于 recipe、计划及发布摘要的规范记录。导入和解析记录不加载 Torch，纯元数据规划或阶段前置证据重验不需要探测当前宿主。

`capture_frozen_backend_precision(device_type)` 在实际模型设备上读取有效数值设置，拒绝非 IEEE 的矩阵运算模式。`require_frozen_backend_precision(expected)` 比较当前设置与冻结记录。`frozen_fp32(expected)` 在完整前向的前后检查合同并关闭 autocast；不写进程全局精度开关。

## 行为边界

冻结推理 recipe 为 `fp32_ieee_single_prefix_v3`，包含具体后端身份。即使参数仍是 FP32，CPU MKLDNN 的 BF16/TF32 或 CUDA 的 TF32 也不能通过冻结入口。绑定后的后端、SDPA 选择或确定性设置改变时拒绝继续计算，不能使用旧校准门槛或缓存。

校准辅助头与单前缀策略前向使用同一边界。显式 FP32 的完整验证也检查 IEEE，未校准的 BF16 训练验证保留原有支持。本合同不改变模型形状、训练损失或允许的 BF16 训练配方；也不承诺不同硬件、库或内核版本之间逐位相同。

旧单前缀 recipe 不能通过改字段/hash 变为当前校准证据。CPU 实际降精度路径及旧发布拒绝回归见[本轮修复记录](../../../kar-nagashi-precision-repair.md)。CUDA 合同分支的结构测试与实际 GPU 验收须分别报告。
