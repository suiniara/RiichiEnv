# KAR / Mahjong Soul: 流局满贯资格合同

## 范围与来源

本合同只明确 KAR 冻结的四人雀魂半庄规则；不将天凤、赛事规则或一般日麻惯例当作雀魂的规则依据。

主要证据是[雀魂日文官方 FAQ](https://mahjongsoul.com/faq/)中 `id=47`、问题“採用役は？”的“流し満貫”条目。官网页面的前端 `index.5518a9a3.js` 以 GET 请求[官方 FAQ 数据](https://mahjongsoul.com/api/faq/list)，2026-11-17 本地只读核验得到：

> 流局時に、自分の捨て牌がすべて么九牌で他家に1枚もポン・チー・カンをされなかった場合に成立する

即：流局时，自己的全部舍牌为幺九牌/字牌，且没有任何舍牌被他家吃、碰、杠，即成立。条文没有附加门前条件。**从该完整成立条件推导**：自己吃、碰、大明杠、暗杠、加杠本身都不取消资格；这五项不是官网逐项列举的独立引文。

辅助证据：[雀魂专门役说明（非官方）](https://nekomimi.ws/~nikukyu/jantama/jt_yakumain.html)明确写道：

> なお、自分が鳴いてしまっても捨て牌さえ揃えば成立する

该辅助页不是官方来源，不取代官方 FAQ。源码和新增测试均只验证实现是否遵守该规则，不能反过来证明平台规则。

本次捕获的官方 FAQ JSON 原字节 SHA256：
`a163b886d4400a2e1c8272b825639c498d5d221fdc08bd11e533da5211896d87`。
辅助页面原字节 SHA256：
`c9f76f6467005c20ff267fbca7694f7e8d2cfbe2e4f2f548f32e5488b5bb3de2`（UTF-16）。

## 资格矩阵

下表的“仍可能”均以自己的已知舍牌没有中张、也没有被叫走为前提；不是保证最终发生。

| 已知事实 | 雀魂目标规则 | KAR 公开掩码 | 独立终局结算 | 原生 live / LogAction replay |
|---|---|---|---|---|
| 无 meld、全幺九字牌、未被叫走 | 可以成立 | 仍可能 | 符合完整条件则计入赢家 | 保留资格 |
| 自己 CHI | 本身不取消 | 仍可能 | 不以自己的 meld 排除 | 保留资格 |
| 自己 PON | 本身不取消 | 仍可能 | 同上 | 保留资格 |
| 自己 DAIMINKAN | 本身不取消 | 仍可能 | 同上 | 保留资格 |
| 自己 ANKAN | 本身不取消 | 仍可能 | 同上 | 保留资格 |
| 自己 KAKAN | 本身不取消 | 仍可能 | 同上 | 保留资格 |
| 自己舍牌被他家 CHI/PON/DAIMINKAN | 该玩家被排除，不是叫牌者被排除 | 完整历史可由历史/河差确认；独立已知中张仍直接排除 | 不计入该玩家 | 清除舍牌者资格 |

MJAI `apply_event` 是另一条事件摄入入口，不是 `LogAction` replay 的别名。本次补齐其 own-middle / called-away 状态更新，使之与 live 和 LogAction 的 `nagashi_eligible` 一致；MJAI 的终局事件仍只是摄入，不改成支付裁判。

## 公开响应掩码与独立结算

掩码是保守的可能集合。单玩家检查：

- 收集已知 `discard_history`、`current_rivers` 的 base34 牌种；只为 actor 加候选。
- 已知中张立即否定。
- 没有否定事实而历史不完整时，保留可能性；不以不完整容器长度差推断叫走。
- 历史完整时保留原 called-away 长度检查。
- 对四名玩家取 any，不修改 snapshot，不读取隐藏状态或真实响应。
- 杠响应窗口仍不能进入普通舍牌掩码入口；pending KAKAN 不提前改成 committed KAKAN。

独立结算仍自行重建四家状态、结构张数、听牌、赢家与支付，不调用公开掩码推导赢家。此次不改变满贯支付、本场、供托或连庄计算。

## 版本与旧证据

- `Ruleset.name` 从 `mahjong_soul_4p_red_half_v1` 更新为 `mahjong_soul_4p_red_half_v2`，使 `ruleset_hash` 明确变化并传播到已有数据、支持、校准和推理绑定；并非引入另一个平台或通用规则框架。
- `RESPONSE_MASK_VERSION` 为 `public_rule_response_mask_v2_mjsoul_nagashi`。
- Token Schema、动作/响应数量、模型、损失与决策数学不变。

旧 v1 导出可能包含错误的 `response_allowed`，尤其是完整且已鸣牌的真实前缀。旧 M0 审计/验收与其派生的 SQLite、事件支持、校准、策略选择及发布证据不能静默标为 v2：需要按新来源重新审计/导出，重建依赖证据或显式迁移。旧权重不因规则标签改变而必然需要重新训练，但旧发布/标定的来源证明不能直接复用。

M1 的 M0 选择入口及 DecisionDataset 会比较当前 ruleset 合同；RuntimeBinding、EventSupport 和 CalibrationArtifact 的已有合同校验随新 hash 拒绝旧绑定；严格发布入口还比较 source/native 指纹。历史报告自身可继续作为历史文件读取，不代表当前兼容。不得手工修改旧 manifest/hash 以伪装重新验证。

此次只报告重建范围，不启动重导出、标定或训练。Rust 源码若改变，已安装的 Python 扩展不会自动更新；源码 Rust 回归与本机既有 Python 二进制回归必须分别报告，不将其混为一次构建验收。

## 回归证据的性质

- `tests/kar/test_observation_response.py`：单玩家/四玩家掩码与部分历史纯函数场景；错误的“所有 meld 都取消资格”预期依上述外部证据修订，保留原场景。
- `tests/kar/test_nagashi_consistency.py`：合成但合法的完整本局，136 物理牌全排列、三张赤五、所有动作由原生引擎实际提供；比较独立结算、公开响应及实际支付。它们不是来自平台的真实牌谱，不冒充真实 Golden。
- `riichienv-core/tests/nagashi_eligibility.rs`：有界事件前缀的 MJAI / LogAction 资格状态比较，不冒充完整对局。
- 原有 Rust `nagashi_mangan` / `nagashi_mangan_replay` 是支付和既有牌谱的兼容回归；其中天凤来源的真实片段不用于证明雀魂的 meld 规则。
