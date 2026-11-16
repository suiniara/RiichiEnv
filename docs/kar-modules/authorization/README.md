<!-- kar-module: riichienv.kar.authorization -->
<!-- api: UpdateAuthorization, ParameterUpdatesNotAuthorizedError, update_implementation_sha256, load_update_authorization, require_parameter_updates -->
# authorization — 参数更新默认拒绝

源码：[authorization.py](../../../src/riichienv/kar/authorization.py) · [总索引](../README.md)

## 职责与接口

- `UpdateAuthorization` 承载明确批准的文档。
- `load_update_authorization(path)` 读取受大小/存在性约束的许可；None 保持未授权。
- `update_implementation_sha256()` 获取当前实现身份。
- `require_parameter_updates(authorization, operation, dataset_hash=None, phase=None, recipe_sha256=None)` 在受维护的更新入口逐层核对。
- `ParameterUpdatesNotAuthorizedError` 表示拒绝，不应被捕获后自动继续训练。

## 放行内容

文档必须有正确版本/内容摘要、明确用户批准与引用、允许的 exact operation、当前源码、真实 readiness 及数据资格/资源状态。提供实际 dataset、phase、recipe 的调用点还要求逐项匹配。不能只在外层检查“某份授权存在”。

例如 GPU 预检先默认拒绝，再打开实际 train manifest 对比绑定，之后才能查询 GPU、前向、backward 或 step。

## 范围与限制

CPU smoke、过拟合诊断、正式训练和 CUDA preflight 中实际发生 backward/step 都算参数更新；不能换成“预检”名称绕过。许可文档不由 README、M0 PASS 或此模块自动生成。

这里保护项目维护的入口，是流程合同，不是任意第三方 Python/Torch 程序的强制安全沙箱。内容 SHA 也不是秘密签名；操作者仍须确保审批文件来源可信。

无独立 CLI，不自动批准、更改权限、联网或运行模型。评测/标定使用另一个 [evaluation_authorization](../evaluation_authorization/README.md)合同。

## 测试与状态

对应 [core/test_authorization.py](../../../tests/kar/core/test_authorization.py)、[production/test_jobs.py](../../../tests/kar/production/test_jobs.py)、[test_review_job_guards.py](../../../tests/kar/production/test_review_job_guards.py)。本轮未创建许可或启动更新。
