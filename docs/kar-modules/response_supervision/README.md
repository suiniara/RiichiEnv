<!-- kar-module: riichienv.kar.response_supervision -->
<!-- api: unique_pass_provenance, ResponseOffer, ResponseWindow, OpportunityCounts -->
# response_supervision — 冻结响应机会与可证动作

源码：[response_supervision.py](../../../src/riichienv/kar/response_supervision.py) · [主设计 §5.3](../../KAR-v1.2.2-DESIGN.md) · [总索引](../README.md)

`ResponseWindow` 持有窗口打开时原生观察、完整全局候选及每席私有标记触发身份。实际鸣牌／荣和标签来自已观察的结果，未成立选择不等于 Pass。`unique_pass_provenance` 只在所有其他候选与已验证裁定矛盾时补 Pass：正常下家摸牌、实际岭上摸牌、严格低优先级的成立鸣牌，或可证明该席未 Ron 且没有其他私有可能的终局。未知及被覆盖意图标为删失，不进入 CE、不生成私有历史。

`OpportunityCounts` 以一席一个非平凡窗口为单位，检查 `total=explicit+uniquely_inferred+censored`，分响应阶段和真实关闭结果汇总；Pass-only 不计数。`GameWriter` 将其与实际发出的监督行核对，manifest 另按 split 汇总。私有标记只在公开锚后追加，不改当前前缀、其他席或公共语义时钟。

[完整半庄回归](../../../tests/kar/test_response_supervision.py)覆盖实际抢杠 Pass、Pon/Pass 同一前缀及无法从公开结果区分的 Chi/Pass 竞争。无独立 CLI，不读取私有客户端输入；统计不是缺失行为被完整识别的证据。
