<!-- kar-module: riichienv.kar.release -->
<!-- api: source_files, write_build_identity, implementation_release_identity, complete_release_contract -->
# release — 源码/native 构建和完整发布合同

源码：[release.py](../../../src/riichienv/kar/release.py) · [总索引](../README.md)

## 接口与身份

`source_files(root)` 对包内 Python/JSON 源文件生成摘要。`write_build_identity(package, code_commit, dirty, source_reference)` 在打包前写一次 `_build_identity.json`；已有身份不静默覆盖。

`implementation_release_identity()` 校验打包身份或本地 Git 版本/dirty 状态，并记录实际 native 扩展 hash、Python 和平台。缺明确来源时拒绝用 unknown commit 替代。

`complete_release_contract(...)` 联合 runtime 合同、训练/数据、rho、规则特征模式、预算、校准、支持、策略和可选配对计划，供生产检查点和发布层绑定。

## 内容边界

源码身份覆盖 `.py/.json`（构建身份文件本身排除）；纯 README 文档不冒充实现版本。它标识实际来源，不保证实现正确或测试已经通过。Git dirty 的源码、native 扩展和 model weights 是不同身份，都要保留。

配置变化、特征模式变化、旧数据选择或外来构建不能通过改写 hash“升级”。打包来源受限时，应显式创建真实 build identity，而非绕过生产校验。

## CLI 与副作用

`stamp-build --package ... --commit ... --source-reference ... [--dirty]` 经顶层 KAR 入口调用。它写构建元数据，不训练或评测，但必须使用真实 commit 与正确 dirty 声明，且目标包目录应明确。

权重打包/受限读取在 [learning/deployment](../learning/deployment/README.md)，原子训练状态在 [production_checkpoint](../learning/production_checkpoint/README.md)。SHA 是完整性摘要，不是对不可信发布者的数字认证。

## 测试与状态

相关回归见 [production/test_training_contracts.py](../../../tests/kar/production/test_training_contracts.py)、[core/test_frozen_inference.py](../../../tests/kar/core/test_frozen_inference.py)。本轮未 stamp/build、查询设备或加载权重。
