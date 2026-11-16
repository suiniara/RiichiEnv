# KAR learning 源码导航

本子包的 28 个 Python 模块（含 `__init__.py`）对应文档见 [learning 模块 README 索引](../../../../docs/kar-modules/learning/README.md)。[KAR 总索引](../../../../docs/kar-modules/README.md)同时说明所有模块的共同合同与授权边界。

入口 `__init__.py` 不自动导入 Torch；按需导入子模块。正式训练使用 `trainer` / `orchestration`，正式评测使用 `simulation`；历史 `train` / `probe` / `gpu_preflight` 不能冒充完整验收，其中部分入口实际含反向传播和参数更新。

文档示例本轮未执行；需要计算、更新或评测时，另行核验数据、来源、预算与明确许可。
