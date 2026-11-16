# KAR 核心运行接口（无训练）

本轮范围是设计审计中的 P01–P12 与 U01。核心运行代码不等于全部 KAR 产品完成，更不自动授予训练许可。实际验收结果以 `kar-core-artifacts/core-20261116/reports/` 中绑定源码的报告为准。

## 运行链

`PublicSnapshot + 完整合法全局 ID → 冻结前缀 → T_A 合法先验 → log p_min 门控 → 全部 S 候选的独立 ACTION 后缀 → 公开响应 mask → T_E / T_R → 悲观 Q → 受限 KL / 证据回退 → 独立 RNG 采样 → 原生动作`

- 546 个全局动作 ID，候选容量 32；不 Top-K，不合并摸切/手切/立直变体。
- 价值重排只改变门控舍牌集合 S 内部的相对概率；S 外逐动作概率和 S 的总质量保持不变。
- `tau=∞` 严格返回同一先验，不执行额外候选前向。
- 默认 `prior_only`，T=1 且明确标记 `unfitted`；不是已校准策略。
- 不存在有效标定、支持或证据时，有限 tau 不会被静默启用。

## 模块边界

| 模块 | 职责 |
|---|---|
| `runtime_contracts.py` | 冻结概率/效用/支持/证据版本，以及模型、推理配置、训练选择、schema、规则和动作绑定 |
| `probability.py` | 完整合法集合上的稳定 log-softmax/softmax；非法概率严格为零 |
| `support.py` | 独立 train game 支持计数、500-game 筛查、21 分支名次效用、Q/Q_LCB/rho_U |
| `calibration.py` | game 固定 cal_fit/cal_check、静态数组温度接口、log 门控、不可变 artifact、validation-fit tau 网格选择 |
| `policy.py` | 质量保持 KL、Rmax 下界、正确 E1/E2、证据覆盖输入、最终完整分布采样 |
| `runtime.py` | 原始 logits 到最终策略的唯一证据/回退协调器和遥测 |
| `learning/inference.py` | 可选 Torch 边界；eval/no_grad 前缀 KV 与全部 S 后缀评分 |
| `live_events.py` | 原生 step 的完整事件增量、公开结算揭示、多荣和合并、终局供托投影 |
| `agent.py` | 观察者代理、多人响应联合提交、固定规则半庄功能 runner |
| `authorization.py` / `readiness.py` | 数据、核心/完整功能、资源与用户更新许可的分离门禁 |

基本 M0 与纯规则模块不依赖 Torch。

## 接入已加载、经核验的冻结模型

```python
from riichienv.kar.agent import KARAgent
from riichienv.kar.learning.inference import FrozenModelPredictor
from riichienv.kar.policy import PolicyOptions
from riichienv.kar.runtime import RuntimePolicy


def make_prior_agent(seat, loaded_model, actual_train_manifest_sha256):
    predictor = FrozenModelPredictor(
        loaded_model,
        actual_train_manifest_sha256,
        use_rule_features=True,
    )
    policy = RuntimePolicy(predictor, options=PolicyOptions(), seed=17 + seat)
    return KARAgent(seat, policy)
```

这里不加载未知 checkpoint、不拟合温度、不创建优化器、不改变模型参数。`FrozenModelPredictor` 的绑定包含实际模型状态、模型配置和特征模式；缓存不允许跨观察者、决策、权重版本或推理配置复用。

### 有限 tau 的边界

1. 支持 artifact 必须绑定模型实际训练选择，按独立 train 游戏计数；499 个 train 加 1 个 validation 不能凑 500。
2. 生产校准从冻结 validation manifest 派生 cal_fit/cal_check。温度和 log 门控只使用 cal_fit；cal_check 只报告。静态测试 fixture 不能当作生产标定结果加载。
3. p_min 使用门控前至少两个完整舍牌动作的样本，概率分母是全部合法动作。排序后取 0-based `n//100`；阈值用 log 表示，保留相等项。
4. `empirical_only` 需要显式选择和可用独立检查；不带数学安全保证。
5. `certified` 需要调用方提供有效的、覆盖本模型/决策/候选集合的误差上界来源、覆盖范围和置信陈述，且正确 E2 为正。模块不提供统一误差界估计器，NLL/Brier 或 500-game 筛查不是证书。
6. 回退到同一模型先验仅停止本次额外价值重排，不保证先验相对于真实行为策略无损。

## 在线事件与隐私

- 代理只取得自己的 Observation 和白名单 PublicSnapshot，不接触模拟器全知状态。
- 聚合器从完整原生 step 的事件增量推进四个观察者账本；不能把最后一个增量当成全场历史。
- 合法公开和牌后，赢家公开手牌从其此前本家账本恢复；不在结算前传播其他玩家暗手。
- 多荣和在同一响应窗口先验证全部映射，再联合 `env.step`；不得顺序执行后丢掉其余赢家。
- 立直保留 reach 与 discard 两步。吃碰后没有额外摸牌，映射为手切；原生 `forced_tedashi` 字段不等同于这个语义判断。
- 暗/加杠声明和提交分开。缺实际公开里宝、未知结算事件或分数链不一致时 fail-closed。
- 达到决策步数上限返回 BLOCKED，不把截断当完整半庄。

不包含雀魂账号连接、网络实战客户端、MCTS 或对手训练。

## 参数更新默认关闭

本项目的 `train_cpu`、`cpu_probe`、`optimizer_update`、GPU `run_preflight` 默认在模型计算/反向/step 前拒绝。显式更新许可还需匹配当前源码、数据选择、资源就绪、完整功能证据与用户批准；本轮不产生有效更新许可。

M0 数据 PASS、核心测试 PASS 和设备空闲均不等于允许训练。该机制保护本项目维护的更新入口，是流程合同，不是任意第三方 Torch 代码的安全沙箱。

既有包含 backward 或参数更新的测试带 `parameter_updates` 标记，默认跳过。常规核心验收不传旧测试 opt-in 开关，并用失败哨兵统计 backward/optimizer.step 调用。

## 当前功能与实验口径

功能验收只允许纯函数、人工静态数组、冻结模型前向和规则对局；不执行真实模型温度拟合、backward、optimizer.step 或 GPU 更新预检。静态 logits 或未训练的微型模型只验证功能，不代表棋力。

正式 CUDA/DDP 训练器、完整验证/评测调度、游戏级置信区间和 M2/M3 接线现已具备代码，见[生产链说明](kar-production-code.md)及[逐模块 README](kar-modules/README.md)。本页保留核心接口边界，不把这些代码的存在或历史核心结果视为当前源码运行验收通过，也不自动启动训练。
