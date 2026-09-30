# Information Fusion 核心方法论与源码

本包以最新上传的协调方法文本为依据，统一为：

**多源证据提取 → 可靠性校准与限权融合 → 延迟价值估计 → 每槽联合资源协调 → 实际部署反馈。**

包含完整 LaTeX 方法正文、Python 标准库实现、JSON 配置、图稿、合成示例、消融与压力诊断。
合成数据仅用于验证方法及代码，不能替代论文中的真实 DRL 实验。

## 1. 直接运行

需要 Python 3.10 或以上。核心算法不依赖 NumPy、PyTorch 或 SciPy。
在解压后的目录运行：

```bash
python run_demo.py
```

该命令生成 1500 槽合成日志：前 300 槽固定归一化，后 1200 槽评价；分别运行 exact 和 AO，
再运行 13 个消融条件及 4 个测量压力条件，最后运行测试与约束审计。
所有结果都标记为 synthetic，保留原日志的实际执行配置与推荐配置。

已有结果可以用下面的命令独立复核：

```bash
python validate_package.py
```

运行自有日志：

```bash
python calibrate_reference.py --input real_stream.csv --through-slot 900 --output real_reference.json
python pipeline.py --input real_stream.csv --reference real_reference.json --config config.json --profiles profiles.json --solver auto --output real_results
```

先在 `config.json` 中将评价槽数和节点数设为实际值，在 `profiles.json` 中替换测得的资源需求、
增益、成本、强度、延迟和质量先验。配置、阈值与超参数应在历史训练/验证部分确定。
归一化脚本只校准状态均值/标准差，不自动测量训练增益或成本。
当前示例数值是可运行的参考参数，并非新的实验测量。

## 2. LaTeX 和图片

| 文件 | 用途 |
|---|---|
| `core_methodology.tex` | 完整方法正文，可导入论文 |
| `system_fusion_model.tex` | 系统、多源融合与资源模型，编辑入口之一 |
| `core_coordination.tex` | 最新上传协调文本的重写版本，编辑入口之二 |
| `main.tex` | 独立 XeLaTeX 检查稿 |
| `methods_preview.pdf` | 已编译的方法正文预览 |
| `references.bib` | 方法片段用到的文献条目 |
| `fig/information_fusion_framework.pdf/.svg/.png` | 原框架图，矢量与 600 dpi 输出 |
| `example_results/synthetic_trace.pdf/.svg/.png` | 本包生成的功能轨迹图，明确标记合成数据 |

编辑两个分节文件后，运行 `python assemble_methodology.py` 合成完整正文。
`build_preview.py` 会在编译前自动合成。正文不指定期刊文档类，投稿时导入论文自身模板，
并合并引用键、公式标签和定理环境。标签延续 `eq:15`、`eq:16`、`eq:17`、`eq:29` 至 `eq:34`；
新增融合与证明公式会改变显示编号，不能把标签中的数字当作最终公式编号。
所需宏包为 `amsmath, amssymb, amsthm, graphicx, natbib`，另定义
`\newtheorem{proposition}{Proposition}`。独立稿已有这些设置。

文字和图中标签使用真实 **Times New Roman**；白底、黑字、黑色线条。
数学符号使用随包提供的 TeX Gyre Termes Math，以覆盖完整数学字符。
Times New Roman 的独立字体文件不随包分发；本机已安装时直接运行：

```bash
python build_preview.py
python plot_demo.py
```

也可以指定本人有权使用的字体目录，该目录须含 `times.ttf`、`timesbd.ttf`、`timesi.ttf`、`timesbi.ttf`：

```bash
python build_preview.py --font-dir /path/to/licensed/times_fonts
python plot_demo.py --font-dir /path/to/licensed/times_fonts
python draw_framework.py --font-dir /path/to/licensed/times_fonts --output fig
```

绘图依赖见 `requirements-visuals.txt`；编译另外需要 XeLaTeX、BibTeX 和列出的宏包。
字体缺失时脚本报错，不悄悄替换字体。可选完整命令：

```bash
python run_demo.py --with-visuals --font-dir /path/to/licensed/times_fonts
```

## 3. 公式与代码对应

`methodology_map.json` 给出逐公式标签、模块与时间语义的机器可读映射。

| 模块 | 方法 |
|---|---|
| `fusion.py` | JSD、状态差异、同执行上下文的外部奖励下降；分块 bootstrap；样本数/新鲜度/预测误差/噪声可靠性；KL 限权投影；分歧与缺失处理 |
| `proxy.py` | 未来恢复基线、已知待部署更新、每配置延迟、切线上界与几何上界 |
| `coordinator.py` | 每节点三类资源约束；触发门控；系数缓存；exact/AO；可选支配剪枝；部署和质量反馈 |
| `pipeline.py` | 流式时序；决策先于反馈；完整窗口从下一槽使用 |
| `live_adapter.py` | 实际 DRL 服务后端协议；执行、部署与不可行时的准入接口 |

核心目标为：

\[
\mathcal U_t(i,j)=W_{j,t}+V_{i,t}D_t^{\mathrm{tr}}g_i
-\lambda_C D_t^{\mathrm{tr}}c_i-\omega(D_t^{\mathrm{tr}}q_i)^2
-\kappa_U\widehat U_tD_t^{\mathrm{tr}}q_i.
\]

默认延迟系数为：

\[
V_{i,t}=e^{-\xi\widehat\Gamma_t}\sum_{h=\delta_i}^{L_t}
e^{-h^0_{t+h|t}}\phi_{i,t}(h).
\]

它是给定预测下恢复质量增量的上界系数。它不是未来真实 DRL 回报的上界。
正文含四条命题和证明：限权解与固定权重下的影响界、延迟增益切线上界、剪枝/AO 性质、
具有统一代理误差假设的单槽决策误差界。
公式中的 gain、cost、intensity 分别对应 `gain`、`cost`、`intensity`，不混用。

## 4. 本次主要修正

1. 将错误的当前 `H_t` 切线改为未来基线 `h0_(t+h|t)`；未知未来更新不被宣称与导数无关。
2. 按训练配置引入部署延迟，得到 `V_i,t`，超出剩余视野的增益记零。
3. 将 `W_t * beta_j` 统一为 `W_j,t`；质量先验只在实际执行该配置后更新。
4. 将 `g_i`、`c_i`、`q_i` 分开，并明确区分推荐训练和实际部署。
5. 将窗口选配置改为每槽决策；融合仍使用完整窗口，策略参数只在实际部署后变化。
6. 5×6 默认 exact；较大空间使用多起点 AO，输出收敛标记，不声称全局最优。
7. 加入可选支配剪枝；在当前负载、固定系数下保证代理最优值不变。
8. 加入 `maximum_source_weight`，防止近零 bootstrap 方差让某一来源占据全部权重；
   来源减少时将有效上限放宽到至少 `1 / 有效来源数`。
9. 复杂度纳入融合/bootstrap、延迟系数、剪枝、初始化和 pair cache。
10. 明确操作变化分数、分歧指数、代理回报和实际环境回报的区别。

## 5. JSON 配置与诊断

- `config.json`：主配置，默认三源、可靠性融合、0.7 单源限权、tangent 代理、auto 求解。
- `profiles.json`：5 个训练配置、6 个推理配置。资源允许三维公共向量或 `M×3` 数组。
- `input_schema.json`：解码后反馈行的数据规范；CSV 用 `capacities_json` 存数组。
- `reference.json`：只由历史前缀得到的均值、标准差、槽号和前缀摘要。
- `ablation_plan.json`：13 个消融条件。单源条件关闭跨来源预测校准。
- `stress_plan.json`：缺失、状态噪声和来源冲突；区间相对于评价开始计数。
- `methodology_map.json`：公式、模块、符号和事件时间映射。
- `solver_audit.json`、`validation_report.json`：求解差距与功能审计结果。

独立重跑消融/压力条件：

```bash
python run_experiments.py
```

`experiment_results/diagnostics.csv` 和 `.json` 记录每个条件的资源违反数、未来信息违反数、
权重/掩码检查、推荐训练比例、分歧和下一窗口性能下降预测误差。
这里的预测目标是观测到的性能下降分数，不是真实漂移标签；Gamma 不是漂移概率。
单源条件的 coverage 按启用来源数计算，不能将不同信息预算的 U 当作同一概率刻度。
当前设置只运行一个合成种子，不从这些误差推断统计显著性或实际收益。

## 6. 接入实际 DRL 系统

使用 `FusionPipeline(..., collect_history=False)`，默认仅保留有限窗口；调用顺序为：

```python
decision = pipeline.decide(slot, training_load, inference_load, capacities,
    future_retentions=causal_retention_forecast,
    pending_gain_forecast=known_pending_deployments)
# 执行 decision；随后获取实际结果，而不是填入推荐配置的虚构结果。
observed = pipeline.observe(actual_feedback)
```

也可实现 `live_adapter.ServiceBackend` 并调用 `run_live`。
后端负责应用动作、真实 DRL 更新、部署队列、请求准入及容量测量。
`pending_gain_forecast[h]` 表示在决策前已知、将在状态 `H_(t+h)` 到达的更新贡献。
`future_retentions[h-1]` 对应 `H_(t+h-1) → H_(t+h)`。
不能使用未来日志中的已观测结果作为预测输入。

反馈中的 `deployed_gain` 必须来自实际部署贡献。
若无法直接测量，可以在真实完成事件后提供 `completed_training_profile` 与
`completed_training_load`，使用明确标记的配置估计；推荐动作本身从不增加 H。
当前可用容量必须已经扣除正在运行任务的占用。
不可行且没有执行配置时设 `execution_status="no_dispatch"`、`logged_training=None`、
`logged_inference=None`；质量估计不更新，但真实待部署贡献仍可记入反馈。
固定日志重放保留日志中的动作与结果，不能测量另一个推荐策略的因果收益。

## 7. 投稿前仍需实证部分

补充真实训练增益与延迟校准；在相同资源、负载和随机种子下执行完整闭环；
评价实际服务回报、SLA 违反、延迟、训练成本、融合诊断和资源协调时间。
在真实漂移标签或明确干预设置下评价检测/预测，并给出多种子区间。
本包验证方法构造和代码一致性，没有重新生成此前论文中的实测收益数值。
