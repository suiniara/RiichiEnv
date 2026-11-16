<!-- kar-module: riichienv.kar.observation -->
<!-- api: PublicMeld, PublicSnapshot, ObserverLedger, project_observer, extract_rule_features -->
# observation — 观察者账本与冻结快照

源码：[observation.py](../../../src/riichienv/kar/observation.py) · [总索引](../README.md)

## 职责

从明确截止点之前的事件建立某一观察者可见的白名单状态。它不是原生全知 GameState 的直接序列化，也不允许把其他玩家暗手、未知牌山或未来响应塞进模型输入。

## 主要接口

- `ObserverLedger(observer_player_id, initial_seat_order_player_ids)`：创建四人观察者账本。
- `ObserverLedger.apply_event(event)` / `snapshot()`：推进一个事实并生成不可变 `PublicSnapshot`。
- `ObserverLedger.check_observation(observation)`：核对本家原生手牌等前缀事实。
- `PublicMeld`：已提交的公开副露；`PublicSnapshot`：本家手牌、公开局况、历史与 pending 状态。
- `project_observer(events, seat, cutoff_event_id, initial_order)`：只投影指定 cutoff。
- `extract_rule_features(snapshot)`：由该公开状态推导确定性特征。

## 核心合同

历史舍牌集合与当前物理河牌分开保存：被叫走的牌仍用于 Genbutsu/Suji，但当前物理计数只能在副露区计一次。固定起家顺序负责同分 rank；当前庄家轮换不改变它。

暗/加杠先保存 pending；加杠仍展示已成立 PON。只有 [kan_transactions](../kan_transactions/README.md)确认当前杠提交才更新副露。旧杠宝牌不能提前提交新杠，抢杠后清除未提交事务。历史位置基于可见逻辑组而非原始索引数值大小。

## 使用与错误

推荐消费正常化事件或 [live_events](../live_events/README.md)的公开事件，不手工拼全知快照。`project_observer` 用于离线 cutoff；实时 agent 维护自己的 ledger。缺起局、重复/不一致立直接受、杠确认不匹配、原生本家手牌不一致或公开墙计数非法时拒绝继续。

## 测试与状态

对应 [test_observation_response.py](../../../tests/kar/test_observation_response.py)、[test_tokenize.py](../../../tests/kar/test_tokenize.py)、[test_review_kan_transactions.py](../../../tests/kar/test_review_kan_transactions.py)。本轮未执行投影示例或规则对局；快照可序列化不代表隐私/因果合同已经重新验收。
