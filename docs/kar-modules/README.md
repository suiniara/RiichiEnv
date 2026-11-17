<!-- kar-module: riichienv.kar -->
<!-- api: Response, Ruleset -->
# KAR 模块 README 总索引

源码：[kar/__init__.py](../../src/riichienv/kar/__init__.py)。本目录为 `src/riichienv/kar/` 的 **73 个 Python 模块**逐一提供说明；本页对应主包 `__init__.py`，[learning 索引](learning/README.md)对应其子包 `__init__.py`，其余文件各有独立目录与 `README.md`。JSON 资源另见[冻结资源说明](assets/README.md)。

KAR 以 **RiichiEnv** 为宿主。`from riichienv.kar import ...` 仅导出 `MJSOUL_RULESET`、`SCHEMA_ID`、`Response`、`Ruleset`，不自动加载 Torch、训练模型或启动对局。完整设计到实现的对应关系见[58项覆盖表](../kar-implementation-coverage.md)，端到端工作流见[生产代码说明](../kar-production-code.md)。

## 共同约束

- 当前 Schema 为 `kar-riichienv-v1`：546 全局动作、最多 32 个完整候选、62 个状态槽；不按候选下标替代动作身份，不 Top-K 截断。
- 主 cohort 固定为雀魂 `category=2 / mode=2 / mode_id=12`；真实 Golden 整场排除在训练、验证、测试和支持统计之外。天凤或其他房间不混入主训练。
- 原始 `.actions` 只读取 `.result`，不读取 `.user_input`。公开输入不得使用他家暗手、隐藏牌山、真实响应标签或未来事件。
- 固定起家顺同分破平；不按当前庄家重新排序。暗/加杠声明与提交分开，旧杠的延迟宝牌不是新杠提交证据。
- 三头输出为 `546 / 21 / 21×4`，训练损失为 `L_A + .25 L_E + .1 L_V`；rho 来自实际训练 manifest。
- cal_fit 用于选择；cal_check 仅作独立诊断；test 不调参。500 个独立 train 游戏是支持筛查，不是误差证书。
- 默认 prior-only；没有有效误差上界不得填零或自动 certified。经验校准与 bootstrap 区间不是逐状态安全证明。
- 数据资格、功能证据、资源就绪和用户授权是不同条件。文档、示例、M0 PASS 均不授予模型更新或评测许可。
- `content_hash` 等 SHA-256 字段用于来源/一致性检查，**不是秘密签名或对恶意发布者的认证**。检查点只从已核验、可信来源加载。

## 阅读顺序与依赖链

1. [contracts](contracts/README.md)、[schema](schema/README.md)、[actions](actions/README.md)：协议与身份。
2. [mjsoul](mjsoul/README.md) → [observation](observation/README.md) → [tokenize](tokenize/README.md)：数据到输入。
3. [audit](audit/README.md) → [corpus_audit](corpus_audit/README.md) → [acceptance](acceptance/README.md) → [export](learning/export/README.md)：可信训练数据。
4. [learning](learning/README.md)：模型、监督、训练、检查点。
5. [calibration](calibration/README.md) → [runtime](runtime/README.md) → [agent](agent/README.md)：校准与在线策略。
6. [planning](planning/README.md) → [simulation](learning/simulation/README.md)：冻结计划、成本证据和独立 M3。

## 基础、数据与规则模块

| 模块 | README | 主要职责 |
|---|---|---|
| `__main__` | [入口](__main__/README.md) | 延迟导入 CLI 分发 |
| `contracts` | [说明](contracts/README.md) | 规则、cohort、split、通用 hash |
| `versioning` | [说明](versioning/README.md) | 设计引用、严格产物版本与半庄计数 |
| `schema` | [说明](schema/README.md) | Token 注册表与二进制校验 |
| `actions` | [说明](actions/README.md) | 原生动作与 546 ID 双向映射 |
| `mjsoul` | [说明](mjsoul/README.md) | 协议加载、result-only 解码 |
| `corpus_fetch` | [说明](corpus_fetch/README.md) | 有界只读搬运 |
| `rare_search` | [说明](rare_search/README.md) | 有界稀有候选定位 |
| `corpus_audit` | [说明](corpus_audit/README.md) | 逐游戏审计与恢复 |
| `audit` | [说明](audit/README.md) | 完整重放协调 |
| `input_audit` | [说明](input_audit/README.md) | cutoff 输入检查与决策回调 |
| `decision` | [说明](decision/README.md) | 不混合输入和标签的决策载体 |
| `acceptance` | [说明](acceptance/README.md) | 单/多分片最终 gate |
| `golden` | [说明](golden/README.md) | 真实来源 Golden 核验 |
| `observation` | [说明](observation/README.md) | 白名单观察者账本 |
| `tokenize` | [说明](tokenize/README.md) | 前缀、ACTION、历史缓存 |
| `rule_features` | [说明](rule_features/README.md) | 确定性公开规则特征 |
| `response` | [说明](response/README.md) | 实际响应窗口标签 |
| `response_mask` | [说明](response_mask/README.md) | 公开规则允许类别 |
| `kan_transactions` | [说明](kan_transactions/README.md) | 公共杠提交判据 |
| `native_replay` | [说明](native_replay/README.md) | 原生重放上下文桥 |
| `payments` | [说明](payments/README.md) | 独立付款与供托 |
| `settlement` | [说明](settlement/README.md) | 独立番符/款项链核验 |
| `completion` | [说明](completion/README.md) | 完整半庄终局证书 |

## 策略、证据与生产合同模块

| 模块 | README | 主要职责 |
|---|---|---|
| `probability` | [说明](probability/README.md) | 稳定合法概率 |
| `support` | [说明](support/README.md) | train-only 支持与保守价值 |
| `policy` | [说明](policy/README.md) | 受限 KL、E1/E2、采样 |
| `runtime` | [说明](runtime/README.md) | logits 到最终策略协调 |
| `runtime_contracts` | [说明](runtime_contracts/README.md) | 模型/数据/推理绑定 |
| `agent` | [说明](agent/README.md) | 在线选择与半庄 runner |
| `live_events` | [说明](live_events/README.md) | 原生增量转公开事件 |
| `calibration` | [说明](calibration/README.md) | 三头温度、log 门控、标定产物 |
| `calibration_roles` | [说明](calibration_roles/README.md) | 不变的 game 级校准角色 |
| `calibration_quality` | [说明](calibration_quality/README.md) | 预登记质量判定 |
| `selection_quality` | [说明](selection_quality/README.md) | 配对验证收益判定 |
| `statistics` | [说明](statistics/README.md) | 自然频率指标、游戏级 CI、直方图 |
| `experiments` | [说明](experiments/README.md) | 固定 10000 paired blocks |
| `simulation_contracts` | [说明](simulation_contracts/README.md) | 成对基准与 M3 成本证据 |
| `planning` | [说明](planning/README.md) | 不加载模型的计划生成 |
| `budget` | [说明](budget/README.md) | 分阶段费用和持久预留 |
| `release` | [说明](release/README.md) | 源码/native 构建与发布身份 |
| `readiness` | [说明](readiness/README.md) | 数据/功能/资源状态 |
| `authorization` | [说明](authorization/README.md) | 参数更新许可 |
| `evaluation_authorization` | [说明](evaluation_authorization/README.md) | 冻结评测许可与互斥锁 |

## learning 子包

所有 28 个 learning 模块（含包入口）见 [learning 模块索引](learning/README.md)。多数学习运行模块需要可选 `kar-ml` 依赖；配置、采样和若干元数据工具不要求加载模型。

## 使用示例与状态

所有命令以 RiichiEnv 仓库根目录为工作目录。仅查看顶层帮助的示例：

```bash
uv run --no-sync python -m riichienv.kar --help
```

含 `train`、`m1`、`m2`、`calibrate`、`select-tau`、`benchmark`、`m3` 的操作有计算或更新副作用，应先阅读对应模块与授权说明。历史 `probe` / `gpu_preflight` 也会执行 backward / optimizer step，不能当作“只读检查”。

**本轮 README 编写没有执行这些示例或测试，没有启动模型/训练/对局，也没有连接远端。** 单个模块存在测试源码不等于当前代码已通过运行验收。历史证据必须核对其数据范围、源码与构建版本。
