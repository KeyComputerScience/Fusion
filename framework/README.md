# 信息融合方法重构与可运行源码

本包依据所提供的系统模型文本和融合框架图重写方法部分。正文使用学术英语；图中文字为 Times New Roman，白底黑字。`methods_reconstructed.tex` 可替换原来的三个方法小节，并增加在线协调小节。`main.tex` 是独立预览入口，投稿时应将方法片段导入论文所用的期刊模板。

新增的技术链路为：**窗口对齐 → 三来源证据提取 → 可靠性评估与自适应加权 → 漂移/分歧输出 → 资源协调 → 延迟反馈校准**。这些内容属于提出的方法扩展；本包提供的是功能参考实现。合成演示不证明论文原有性能结果，也不构成录用概率的依据。

## 文件

| 文件 | 内容 |
|---|---|
| `methods_reconstructed.tex` | 重构后的英文方法正文、公式、图注 |
| `main.tex` / `methods_preview.pdf` | 独立编译入口与已编译预览 |
| `references.bib` | 核验过的 ORRIC、RECL 与概念漂移综述条目 |
| `fusion.py` | JSD、状态变化、匹配上下文的服务退化、块自助法、可靠性、权重与反馈校准 |
| `coordinator.py` | 多节点资源约束、动态服务质量、延迟恢复价值、精确搜索与多起点 AO |
| `pipeline.py` | `decide()` / `observe()` 流式接口及离线日志回放 |
| `calibrate_reference.py` | 仅使用评估之前的历史前缀拟合标准化参数 |
| `generate_demo.py` | 固定随机种子的合成日志生成器 |
| `config.json` | 窗口、可靠性、触发、代价与求解器参数 |
| `profiles.json` | 5 个训练和 6 个推理配置；资源与代价需替换为实测值 |
| `reference.json` | 演示用冻结统计量及前缀摘要 |
| `input_schema.json` | 解码后单行日志的 JSON Schema |
| `draw_framework.py` | 该框架图的矢量绘图源码：PDF / SVG / 600 dpi PNG |
| `plot_demo.py` | 演示轨迹绘图脚本：白底黑字、Times New Roman |
| `build_preview.py` | XeLaTeX / BibTeX 编译脚本 |
| `tests/test_core.py` | 因果时序、资源可行性、缺失来源等验证 |
| `validate_package.py` | 重跑测试，并核对两种求解器的约束、时序及代理目标差距 |
| `example_stream.csv` / `example_results/` | 合成输入、决策、融合记录及运行摘要 |
| `validation_report.json` | 本次运行的检查结果 |

## 快速运行

核心算法仅依赖 Python 3.10+ 标准库。解压后在本目录执行：

```bash
python3 generate_demo.py
python3 calibrate_reference.py --through-slot 300
python3 pipeline.py --solver auto --output example_results
python3 pipeline.py --solver ao --output example_results_ao
python3 -m unittest discover -s tests -v
python3 validate_package.py
```

默认生成 300 个校准槽和 1200 个评估槽，窗口长度为 50。默认 5×6 配置由精确搜索处理；`--solver ao` 可显式检验 AO。日志从全局槽 301 开始评估，论文中的评估槽 1 对应该槽。首次有效相邻窗口对在全局槽 400 结束后产生，并从槽 401 起可用。两种求解器均优化当前代理目标，AO 的结果不保证全局最优。

输出包括 `decisions.csv`、`fusion_windows.jsonl` 和 `summary.json`。每个决策记录包含其采用的已完成窗口末槽、融合权重、漂移、缺失/分歧指标、推荐配置及实际部署反馈。回放保持日志中真正执行的配置与推荐配置分开，不能用既有日志的奖励评价另一组推荐动作。

### 编译与绘图

LaTeX 预览需要 XeLaTeX、BibTeX 和 Times New Roman。字体已安装时：

```bash
python3 build_preview.py
```

字体以文件形式提供时，目录应包含 `times.ttf`、`timesbd.ttf`、`timesi.ttf`、`timesbi.ttf`：

```bash
python3 build_preview.py --font-dir "/path/to/licensed/fonts"
python3 draw_framework.py --font-dir "/path/to/licensed/fonts" --output fig
python3 plot_demo.py --font-dir "/path/to/licensed/fonts"
```

绘图的可选 Python 依赖见 `requirements-visuals.txt`。包内不附带独立 Times New Roman 字体文件。正文和图中文字使用实际 Times New Roman；数学符号在预览中使用附带的开源 TeX Gyre Termes Math，其许可和来源说明位于 `fonts/`。已有框架图 PDF/SVG 嵌入字体，可直接引用。预览 PDF 是方法部分的排版检查件，完整投稿稿件仍应使用期刊模板并补齐作者、摘要、实验和其他章节。

## 公式与代码对应

| LaTeX 标签 | 定义 | 代码位置 |
|---|---|---|
| `eq:fusion-workload` | 归一化 JSD | `normalized_jsd()` |
| `eq:fusion-operating` | 冻结参考尺度下的状态均值变化 | `EvidenceFusion._evidence()` |
| `eq:fusion-performance` | 相同训练/推理配置与策略版本下的奖励退化 | `EvidenceFusion._performance()` |
| `eq:fusion-reliability` | 支持度、时效、噪声与预测误差可靠性 | `complete_window()` |
| `eq:fusion-calibration` | 下一窗口外部退化反馈 | `PredictiveCalibration` |
| `eq:6` / `eq:fusion-uncertainty` | 加权漂移、分歧、缺失覆盖率 | `FusionSnapshot` |
| `eq:9` | 每节点、每资源可行性 | `Coordinator.feasible()` |
| `eq:11` | 已部署更新驱动的恢复状态 | `Coordinator.observe()` |
| `eq:12` | 已执行配置的服务质量更新 | `service_quality()` / `observe()` |
| `eq:15` | 滞回触发与分歧/覆盖率准入 | `Coordinator._allowed()` |
| `eq:16` / `eq:17` | 有限前瞻下的延迟价值与配置目标 | `Coordinator.value()` / `choose()` |

保留了原文 `eq:1` 至 `eq:14` 和 `fig:2` 标签，补齐原来引用但未定义的 `eq:15`。增加公式后，实际显示编号会变化；全文应统一改为 `\eqref{...}`，避免硬编码“(13)”之类编号。参考文献保留键 `12`、`13`，新综述使用 `drift_review`；合并至完整 `.bib` 前核对已有条目，避免重复或键冲突。

## 接入真实数据与闭环服务

1. 使用评估前的数据拟合 `reference.json`，固定训练/验证/测试时间切分。任务到 `regime` 的映射应先定义并冻结，不能根据评估结果重新分桶。
2. 测量各配置在每个节点的 memory/compute/budget 单位工作量资源需求。`profiles.json` 中三元组广播到所有节点；也支持 `M×3` 的节点异构矩阵。容量采用相同单位。
3. 单独校准恢复增益、直接成本、更新强度、部署延迟和质量先验。`gain`、`cost` 与 `intensity` 不可当作同一量。示例数值只服务于接口演示。
4. 外部服务奖励写入 `environment_reward`。仅当该奖励确实扣除了已知训练成本时，设置 `observed_cost_reward_scale` 补回该成本；否则设为 0。不能将 `Coordinator.value()` 或理论 `R_model` 写入该列。
5. 用 `FusionPipeline.decide(slot, training_load, inference_load, capacities)` 获取当前决策；执行后调用 `observe(feedback)`。观察行格式见 `input_schema.json`。在线核心不实现 DQN/PPO 网络训练，服务策略的采样、优化器更新、模型替换由现有 DRL 服务接口执行。
6. `deployed_gain` 必须对应已完成部署的更新。如只有已完成配置与工作量，可提供 `completed_training_profile` 和 `completed_training_load`，代码会使用配置增益估计并明确标记；缺少两类信息时按增益 0 处理并记录这一假设。建议记录更新任务 ID、启动槽和部署槽，以核查真实延迟。
7. 缺失来源用空 CSV 单元格或 JSON `null` 表示。没有共同策略版本的奖励窗口不强行比较。全部来源缺失时保持最近漂移值并禁止训练准入；资源不可行时返回 `infeasible_admission_required`，由上层决定延后或拒绝请求。

可靠性反馈使用“上一窗口源分数预测下一窗口外部退化”的已观察误差；它衡量预测效用，不是来源真实性的证明。状态与服务指标仍可能受策略影响，匹配上下文无法消除全部混杂。分歧指标也不是统计置信区间。

实证稿建议报告外部回报、完成率、截止期违约、资源消耗、更新部署延迟、求解时间，以及各来源缺失时的表现。至少应比较单源、固定权重、可靠性加权、去除分歧处理、去除反馈校准等消融。真实闭环实验完成前，应使用“proposed / reference implementation”的表述，避免将本包的合成结果写成已验证的论文实验结论。
