# Performance Analysis

## 本次已执行的实验

| 项目 | 数量或设置 |
|---|---|
| 系统 | 10 个服务节点、1 个逻辑云控制器 |
| 校准前缀 | 每个种子/骨干 1200 slots；不计入测试收益 |
| 测试 | 每次 3000 slots，窗口 50，共 60 个完整窗口 |
| 配对种子 | 10、20、30、40、50 |
| 主对比 | DQN/PPO × 5 种协调策略 × 5 seeds = 50 次 |
| 闭环消融 | DQN × 9 种消融 × 5 seeds = 45 次 |
| 融合诊断 | 125 组源证据对比，使用相同的 5 份 DQN NoRT 已执行反馈 |
| 求解器对比 | 500 个相同状态；25×30=750 对插值配置 |
| 日志核对 | 285000 slots、531 次实际参数部署；资源违规 0 |
| 测试 | 27 项核心检查 + 8 项闭环检查 |
| 完整重执行核对 | seed 10 的 DQN/PPO Fusion，各 3000 slots；收益及模型参数摘要一致 |

执行环境为 Linux x86-64 / Intel Xeon Platinum 8573C、Python 3.12.14、NumPy 1.26.4、PyTorch 2.3.1+cpu，CPU 单线程。源码声明确定性 Torch 运算。

## 主要结果及解释边界

| 骨干 | Reliability fusion 收益 | No retraining 收益 |
|---|---:|---:|
| DQN | 1905.93 ± 63.50 | 1907.18 ± 63.82 |
| PPO | 1918.98 ± 36.01 | 1921.90 ± 30.88 |

收益是 3000 个测试时隙的外部奖励之和，± 表示五个种子的样本标准差。融合方案未提高平均闭环收益。其共同目标窗口上的预测 MAE 为 0.0475，persistence 为 0.0490；在两个突变目标窗口上，融合 MAE 为 0.3153，persistence 为 0.3108。不能用整体误差的小幅降低宣称可提前预测突变。

AO 的平均代理目标差距为 0.01814，最大差距 0.15292；相同最优配置比例为 40.8%。本次合计耗时比约为 3.96，但 AO 的坐标收敛不等于全局最优。默认 30 对配置使用精确枚举。

五个配对种子的双侧 signed-rank 检验最小可达 p=0.0625。本稿不声称 p<0.05。Bootstrap 区间、完整 seed 数据、次级诊断和负结果均保留。强突变、窗口对齐、有限训练和固定恢复先验限制了外推。

## 安装与一条命令复现

在 Python 3.12 的独立环境中安装：

### bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

Windows 可用 Python 3.12 创建虚拟环境，并用 `.venv\Scripts\Activate.ps1` 激活。论文表中的实际计时来自本包记录的 Linux 环境。

核对已附数据和日志，并完整重跑两个参考实验：

### bash
python reproduce.py --verify-only

生成全部 95 次闭环实验、125 组融合诊断、500 个求解器状态：

### bash
python reproduce.py

快速检查会写入独立的 `smoke_data/` 与 `smoke_results/`，不会作为论文结果：

### `bash
python reproduce.py --smoke

默认完整流程依次执行测试、数据生成、前缀校准、闭环运行、融合诊断、求解器比较、逐条核对、参考重执行、预测导出和 LaTeX 生成。`run_experiments.py` 会拒绝与当前参数不一致的缓存合成数据。

## 文件与生成关系

| 文件 | 内容 |
|---|---|
| `experiment_config.json` | 场景、种子、DRL、外部奖励、基线、消融与扰动参数 |
| `core/config.json`, `core/profiles.json` | 融合/协调参数和完整资源配置向量 |
| `data_schema.json` | 数组形状、单位、日志列和目标可用时间 |
| `data_pipeline.py` | 合成数据生成、Alibaba CSV 流式导入及来源记录 |
| `backend.py` | 排队服务、资源占用、训练暂停与实际延迟部署 |
| `policies.py` | 可执行 DQN 与 PPO；服务动作数为 3 |
| `closed_loop.py` | 校准、决策、执行、反馈、部署与融合调用 |
| `run_experiments.py` | 配对运行及逐时隙输出 |
| `fusion_benchmarks.py` | 同反馈融合诊断和 exact/AO 比较 |
| `analyze_results.py` | 日志验证、配对统计和共同预测目标筛选 |
| `build_performance.py` | 由验证后的分析文件生成文字和表格 |
| `export_data.py` | 数据来源及实际闭环预测例子导出 |
| `verify_reexecution.py` | 两个完整参考运行的收益和模型摘要比较 |
| `repair_logs.py` | 中断日志重生成；要求收益、任务计数和模型摘要完全一致 |
| `core/tests/`, `tests/` | 核心和真实闭环行为的可执行检查 |
| `core/generate_demo.py` | 核心测试使用的小型夹具，不用于本稿实验 |

`results/calibration/` 保存每个骨干/种子的规范化、质量先验、初始策略 `.pt` 和前缀日志。

`results/runs/<backbone>_<method>_seed_<seed>/` 保存 `slots.csv.gz`、`windows.json`、`deployments.json`、`summary.json`。每个完整闭环运行有 3000 行日志。`training_profile` 是本时隙新选择的配置，`logged_training` 是实际在占用资源的训练配置，二者可能不同；`inference_profile` 必须等于 `logged_inference`。

`results/seed_results.csv`、`results/main_summary.csv`、`results/paired_comparisons.csv` 保存原始和汇总数值。`results/analysis.json` 与 `results_for_manuscript.json` 保存完整统计。`results/validation.json` 和 `results/reexecution_checks.json` 记录检查结果。`results/forecast_examples.json` 保留突变附近的已存预测、事后目标和误差，明确目标未进入当时决策。

`artifact_hashes.json` 覆盖归档中的源码与结果文件；`results/run_manifest.json` 记录已执行协议和运行环境。

## 闭环实现的关键约定

- 两层决策：协调器选择训练/推理配置；DRL 选择三类任务服务优先级。
- 外部奖励只使用完成、截止时间损失、排队和真实占用成本，不直接使用 Γ、H 或协调代理目标。
- 私有 worker 在启动时计算梯度，但服务参数只在分配时隙数满足后部署。延迟表示仿真中的占用，不是 GPU 训练耗时。
- 每个时隙先决策后收集反馈；完整窗口快照在下一时隙生效。初始前缀不含测试反馈。
- `g_i` 为固定恢复先验，实际训练完成时才记入 H；它没有被当作实际神经网络收益。
- DQN 前缀含 50 个优化步骤，PPO 前缀含 36 个优化步骤；未以收敛策略作为已验证假设。
- 漂移覆盖可由持续报警贡献，需与非突变窗口报警比例一起看。性能证据缺少相同配置/版本上下文时标为缺失。
- 同一场景内的预测 MAE 使用各比较方法共同可用的目标窗口，避免因缺失样本选择不同而产生偏差。
- 所有源缺失时保持最近有效 Γ，同时标记 fallback、U=1、coverage=0；不能把缺失当作环境稳定。

## Alibaba 导入路线

在提供官方原始文件后运行，支持普通 CSV 或 `.csv.gz`：

### bash
python data_pipeline.py import-alibaba --machine-usage /path/machine_usage.csv --batch-task /path/batch_task.csv --slot-seconds 60 --output real_input
python data_pipeline.py generate --trace real_input/trace.npz --output real_data
python run_experiments.py --data real_data --output real_results

导入器两遍流式扫描 usage：用校准前缀覆盖率选节点，再只聚合选中节点；不会把所有机器的完整时间跨度保存在内存中。有效性检查、原始文件 SHA-256、选中机器、缺失与填充记录都保留。预计需要完整 4200 个聚合槽，约 70 小时输入。任务 CPU 用 100=1 core；内存和利用率百分比除以 100。需求类型分组、下采样到 10 个服务节点、Poisson 请求、预算与服务工作量仍是建模假设。

真实容量可能不支持当前所有配置。校准器会拒绝没有有效支持的推理质量先验，应独立校准真实 trace 的资源需求及任务语义。

