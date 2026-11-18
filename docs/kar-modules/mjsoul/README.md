<!-- kar-module: riichienv.kar.mjsoul -->
<!-- api: Protocol, ServerRecord, DecodedGame, ProtocolError, RecordError, load_protocol, decode_record_bytes, normalize_records -->
# mjsoul — 实际雀魂协议与结果解码

源码：[mjsoul.py](../../../src/riichienv/kar/mjsoul.py) · [总索引](../README.md)

## 职责与接口

- `load_protocol(proto_path)`：由明确的 `.proto` 构造动态描述符，返回 `Protocol`，分别记录源 proto 与 descriptor 身份。
- `decode_record_bytes(data, protocol, container=...)`：验证容器及服务端记录，返回 `DecodedGame`。
- `ServerRecord`：原始记录索引/类型及消息载体；`normalize_records(...)`：将结果记录转成标准事件。source record编号只作来源，不设置模型时钟。
- 九种九牌读取RecordLiuJu的type/seat/tiles，与当前首次draw、未舍牌未鸣牌、14张幺九字条件及先前已验证手牌核对；仅揭示宣布者，不读取allplayertiles补其他手牌。源tiles缺失只允许规则证明的唯一重建并记录provenance。
- 途中流局的零转移由冻结规则证明；已有scores与零deltas生成结算Token。缺失NoTile/Hule结算仍保持缺失，不从后续分数反推或填零。
- `ProtocolError` 表示协议描述符/字段合同问题；`RecordError` 表示原始字节、容器或记录事实非法。

## 容器与数据合同

支持 `res_game_record`（函数缺省值）、`majsoul_download`、`wrapper`、`game_detail_records`。下载归档要求经核验的三零字节前缀，不能靠尝试剥头“修复”未知文件。

`GameDetailRecords.actions` **只消费 `.result`**；`.user_input` 是不透明跳过字段。读取 header/最终分数用于来源和独立比较，不从服务器番符/支付标签反推独立计算结果。

```python
from riichienv.kar.mjsoul import load_protocol, decode_record_bytes

def decode_verified_archive(proto_path, raw_bytes):
    protocol = load_protocol(proto_path)
    return decode_record_bytes(raw_bytes, protocol, container="majsoul_download")
```

## 调用关系与安全边界

上游为显式 raw 文件/本地搬运清单；下游是 `audit`、`corpus_audit`、`golden` 和 `learning/export`。原始 hash、规范化事件和协议 hash 各自保存，不互相替代。不会执行下载的 pb2；相关生成模块比较只允许静态描述符检查。

协议错误、重复/缺失必需字段、未知容器或不可规范化事件应阻断，不自动补造结束事件或猜测隐藏信息。需要 protobuf / proto 编译工具依赖，但不需要 Torch。

## 测试与状态

对应 [test_mjsoul.py](../../../tests/kar/test_mjsoul.py)、[test_audit.py](../../../tests/kar/test_audit.py)。本轮未全量重导出真实牌谱或执行下载模块；合成protobuf/原生合法局面及既有有限只读原始资产的实际结果见[修复报告](../../kar-semantic-branch-repair.md)。外部输入仍遵守隔离目录与只读要求。
