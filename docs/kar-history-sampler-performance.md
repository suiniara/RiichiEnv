# KAR 性能修复：冻结历史缓存、采样摘要及索引内存

审查与实测基线 `cd77d614f473b4a1cfdb618d1bac0931132e2480`。开工时 `suiniara/RiichiEnv` 的 `kar/riichienv-v1-20261117` 远端与本地 HEAD 相同，相关文件相对审查基线无后续提交差异。本地既有 `docs/kar-modules/observation/README.md` 保持原字节；未提交或推送。未发现用户列出的审查脚本，以下为实际运行的本地有限合成复现。

## 1. 改动与合同

| 根因 | 修复位置与原因 |
|---|---|
| 冻结 predictor 在每次 `policy()` 新建历史缓存 | [inference.py](../src/riichienv/kar/learning/inference.py) 为四个观察席各保留一个增量 `TokenHistoryCache`，由实际 `encode_prefix()` 使用。[observation.py](../src/riichienv/kar/observation.py) 提供半庄账本的进程内身份，不写到序列化 snapshot、公共事件、Token 或模型输入。predictor 用该身份更换各席缓存，所以完全相同的开局也不会跨半庄复用。跨小局保持同一账本。历史缩短、改旧事件、逻辑位置变化或迟到 `SELF_PASS` 插入仍由原 [TokenHistoryCache](../src/riichienv/kar/tokenize.py) 检查触发重建；通用缓存文件和原有行为不改。没有账本身份的快照安全地按冷路径编码。既有冻结权重及单决策 K/V 校验保持，并额外拒绝相同 context 和相同可见字节但不同账本的后缀。 |
| 每次检查点重扫完整数据长度和排列 | [sampling.py](../src/riichienv/kar/learning/sampling.py) 将长度固定为只读 tuple、初始化计算摘要，并把各 epoch 的只读排列与摘要同时生成。`state_dict()` 只读摘要；[trainer.py](../src/riichienv/kar/learning/trainer.py) 的 epoch 记录使用同一摘要，不再额外哈希。恢复仍核对当前数据摘要、重新生成顺序、计算并比较摘要、验证游标及更新边界，校验通过才替换状态。 |
| 静态校准逐行扫描游戏 tuple | [calibration.py](../src/riichienv/kar/calibration.py) 仅在 `split.validate()` 之后将两分区转换为临时成员集合；不改变记录及顺序、重复 ID 拒绝、摘要、温度、log p_min 或 cal_fit/cal_check 角色。 |

新增 [真实冻结 predictor 缓存回归](../tests/kar/core/test_frozen_history_cache.py)、[采样/检查点回归](../tests/kar/production/test_sampler_digest_cache.py)，扩展 [校准边界](../tests/kar/core/test_calibration.py)。模块说明见 [inference](kar-modules/learning/inference/README.md) 与 [sampling](kar-modules/learning/sampling/README.md)。

Token Schema、FP32 推理 recipe、数据与采样记录、全局 batch、动作／响应编码、固定 rho、损失、公开响应窗口及删失解释不变。本次不重建数据、不换索引格式，不改变授权和预算门禁。生产发布或精确恢复仍按原有源码、原生扩展、数据与 recipe 绑定验证；不把 sampler 字段兼容当成所有旧发布产物可直接跨源码恢复。

## 2. 编码与三头一致性

对照使用同一原生种子 13、完整合法 136 张墙、12 小局半庄、1718 个公开事件、观察者 0 的全部 210 个本家动作快照。原生真实终局与投影分数一致。分别从连续快照选取接近 1012/1997 Token、相隔恰好 8 条可见事件的两对；每次先用同一观测席的前一张快照暖缓存，再测后一张。两个代码版本都走实际 `FrozenModelPredictor.policy()`；只测 `encode_prefix`，不计 Tensor collation、snapshot 摘要及 Transformer 前向。一次未测前快照 + 一次计时，重复 11 次，CPU 单 Torch 线程。

| 样本 | Token 数 | 每次处理事件数（基线 → 修复） | 编码中位数（基线 → 修复） |
|---|---:|---:|---:|
| 中盘 | 1003 | 878 → 8 | 75.2340 → 7.1581 ms |
| 末盘 | 1947 | 1756 → 8 | 142.8704 → 7.5372 ms |

上述两份实际可见 snapshot 序列化、前缀字节 SHA、完整候选，以及同一冻结 miniature FP32 模型的 Policy／Response／Rank 输出摘要，均与基线完全一致；每次的事件数和原始样本在证据。固定模型的跨观察席、跨小局、相同开局换半庄、真实迟到私有 Pass、旧前缀编辑和历史回退另有逐字节及三头精确对拍。这是编码阶段实测，**不是完整推理提速**。

## 3. 采样检查点开销

CPython 本机合成长度 `449 + i % 819`，相同种子、分桶和全局 batch 参数。计时未启用 `tracemalloc`，排除构建、shuffle、checkpoint 写盘及哈希监控。每规模 11 次，旧每次串行调用 1 次；新路径每次批量调用 4096 次再除以调用数，以免短调用受计时器粒度支配。独立非计时调用检查哈希次数；两个规模的全部序列摘要、state 字段及首组 rank 样本完全相同。

| 决策条数 | `state_dict()` 中位数（旧 → 新） | 非计时探针的全量哈希 |
|---|---:|---|
| 100,000 | 5.7352 ms → 0.197 µs | 两份十万条序列 → 0 次 |
| 1,000,000 | 74.7207 ms → 0.184 µs | 两份百万条序列 → 0 次 |

不把这组数字当作检查点写盘或整体训练提速。实际新增的无更新测试验证了跨 epoch 的原始次序、尾 batch、分 rank 划分、保存/读取生产 checkpoint 后的下一组，以及篡改摘要、非法游标和源码身份不符均拒绝。

## 4. 索引内存预算

实际训练路径 [dataset.py:92](../src/riichienv/kar/learning/dataset.py:92) 通过 SQLite `fetchall()` 物化 `(row_id, history_rows)`；[trainer.py:390](../src/riichienv/kar/learning/trainer.py:390) 为训练数据再构造全长序列，每个 DDP rank 都从自己的完整 train/validation 索引启动。合成测量使用同一 JOIN／ORDER BY 选择、60 ASCII 字符行 ID 和范围大于小整数缓存的 history_rows；以 Python `tracemalloc` 在独立进程测持有值与创建峰值，不同时取任何计时。不是全进程 RSS，也不包括 validation 索引、manifest、SQLite 原生页及排序内存、历史缓存、模型、优化器与分配器开销。

| 训练决策数 | 索引 | 索引 + trainer lengths | 再加 sampler（持有） | 创建 sampler 时的跟踪峰值 |
|---|---:|---:|---:|---:|
| 100,000 | 19.552 MiB | 23.368 MiB | 27.939 MiB | 29.063 MiB |
| 1,000,000 | 195.932 MiB | 234.507 MiB | 280.277 MiB | 293.417 MiB |

百万样本在该 CPython 3.14.9 / SQLite 3.53.1 夹具上约 **205.45 B/条仅索引**、**293.89 B/条含训练 lengths/sampler**。采样 API 生成摘要时有短暂额外内存，而缓存后的单次 `state_dict()` 相对持有状态只额外追踪到 440 B。下面按百万行测量斜率线性外推，未执行更大规模分配；均为训练索引相关持有量，不含其它对象：

| 假设训练决策数 | 1 个 rank | 4 个 rank | 8 个 rank |
|---|---:|---:|---:|
| 1,000,000 | 0.274 GiB | 1.095 GiB | 2.190 GiB |
| 5,000,000 | 1.369 GiB | 5.474 GiB | 10.948 GiB |
| 10,000,000 | 2.737 GiB | 10.948 GiB | 21.897 GiB |

`R` 个本机 rank、`T` 条训练、`V` 条验证时，只估算这部分的公式为 `R × (293.89 × T + 205.45 × V)` 字节；预算还需加齐上述未测项并留创建峰值余量。这些都是合成测量和情景外推，没有现行生产任务的决策数、rank 分布或目标主机可用内存证据，也没有观察到 OOM。因此**保留现有索引结构**，不实施紧凑索引或 mmap。

## 5. 校准与回归结果

相同合成 logits 的整个前后校准产物 SHA 均为 `8532d29cefae9ed12d18c97c5e60699cb7dc4fb91a54c24cf5afe622e76aaa3c`，分区顺序和行摘要不变。改变 cal_check 的标签及 logits 后，cal_check 证据确实变化，cal_fit 的三个温度及 log p_min 不变。默认套件复核外来游戏、错分区和重复 decision identity 拒绝。

| 实际 pytest 运行 | 通过 | 失败 | 错误 | 跳过 |
|---|---:|---:|---:|---:|
| 审查基线上专门复现两个缺陷 | 0 | 2 | 0 | 0 |
| 首轮全套（中间实现，不计入最终） | 2933 | 1 | 0 | 19 |
| **最终默认 `tests/kar`** | **2934** | **0** | **0** | **19** |

首轮失败是 `test_hidden_and_future_source_fields_do_not_invalidate_public_history`：当时在通用缓存中加入了账本身份，导致不同账本构造的同一可见前缀重编。最终把身份隔离放到 predictor 入口，保留通用缓存及原测试断言，重跑通过。最终 19 项跳过包括 15 个需要显式参数更新许可的用例、4 个过期的本地私有导出；均不计 PASS。两个真实本机 CPU/Gloo 验证（报告传播及 primary 错误协调）均实际通过，未遇到审查环境的 `Operation not permitted`，没有新增跳过。父进程禁止 backward、grad、optimizer step 和 GPU 初始化的守卫均为零。Ruff、Ty 和 Git 空白检查通过。最终源码指纹为 `d64a2ca9ee15c84c70b5af668906dbc0526453eaa99a9d4c475aa4207557612b`。

## 6. 复现与证据

证据包提供 `verify.py`、`profile_token_cache.py`、`measure_index_sampler.py`、实际命令、JSON 样本、JUnit 与源码指纹。测试用隔离路径中事先实编的 CPU 原生扩展，Rust 源字节与本基线匹配；未替换本机安装。未启动正式 M1/M2/M3、parameter updates、GPU、真实训练数据重建或付费资源。

复现时将三个脚本放在 `D:\Workspace\kar-fix-reports` 下的一个新空目录，从该目录执行。`verify.py stage` 读取 Git 基线和当前源码，并按已有构建证据使用匹配的原生扩展；脚本拒绝覆盖旧证据。逐项执行：

```powershell
& 'D:\Workspace\RiichiEnv\.venv\Scripts\python.exe' -I -X utf8 '.\verify.py' stage baseline baseline
```

```powershell
& 'D:\Workspace\RiichiEnv\.venv\Scripts\python.exe' -I -X utf8 '.\verify.py' stage final working
```

```powershell
& 'D:\Workspace\RiichiEnv\.venv\Scripts\python.exe' -I -X utf8 '.\profile_token_cache.py' baseline 11 history-baseline-final.json
```

```powershell
& 'D:\Workspace\RiichiEnv\.venv\Scripts\python.exe' -I -X utf8 '.\profile_token_cache.py' final 11 history-final.json
```

```powershell
& 'D:\Workspace\RiichiEnv\.venv\Scripts\python.exe' -I -X utf8 '.\measure_index_sampler.py' baseline timing sampler-baseline-final.json 11 1
```

```powershell
& 'D:\Workspace\RiichiEnv\.venv\Scripts\python.exe' -I -X utf8 '.\measure_index_sampler.py' final timing sampler-final.json 11 4096
```

```powershell
& 'D:\Workspace\RiichiEnv\.venv\Scripts\python.exe' -I -X utf8 '.\measure_index_sampler.py' final memory indexes-memory-final.json
```

```powershell
& 'D:\Workspace\RiichiEnv\.venv\Scripts\python.exe' -I -X utf8 '.\verify.py' run final default default-final-v2
```
