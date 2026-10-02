# DA-RF 方法与 Python 参考实现

`risk_rf_revision.tex` 是完整、可独立编译的 Methodology 源码。当前编辑器继续使用该文件；Python 实现采用 NumPy 和标准库，支持 Python 3.10 及以上。

## 运行

在本目录执行：

```sh
python3 -m pip install numpy
python3 test_da_rf.py
python3 run_da_rf_demo.py --out-dir .
python3 verify_joint_error.py --three-regime-only --seeds 20 --samples 100000
```

Python 路径也可替换为已安装 NumPy 的环境。无需 SciPy、PyTorch 或终端 LaTeX 安装。示例参数用于检验机制，不是经过真实服务数据验证的最佳参数。

## 文件与方法对应

| 文件 | 实现内容 |
|---|---|
| `da_rf_fusion.py` | 质量参考、按可用集合复用的 PSD 残差档案、决策加权矩阵、经验缓冲半径、受约束凸融合、求解证书及敏感性界 |
| `da_rf_predictor.py` | 各源同目标的延迟标签岭预测器，保存发行时的特征和预测，按原始时间衰减迟到标签 |
| `da_rf_evidence.py` | JSD、运行状态变化、上下文匹配服务变化、持续损失目标及保留缺失位置的区块 bootstrap |
| `da_rf_coordinator.py` | 候选相关不确定性、资源可行性、持续损失与滞回、经验更新准入、条件证书标记和预算探测 |
| `test_da_rf.py` | 时间顺序、缺失、PSD、梯度更新、准入、费用和求解行为的回归验证 |
| `run_da_rf_demo.py` | 在线融合流程和单独的真实梯度恢复诊断，保存逐窗口记录与汇总 |
| `verify_joint_error.py` | 同边际误差、同矩阵迹与对角线的固定权重机制验证 |

## 核心创新及边界

1. **按可用集合复用联合误差。** 每条记录保存当时真正观测到的预测掩码。当前集合 A 只读取原掩码包含 A 的记录，并只计算 A 内共同观测的外积。某个源缺失时，其余源仍能学习联合误差；缺失残差不填零。`exact_mask=True` 仅使用完全相同的掩码，作为减少观察分布混合的对照。
2. **决策收益对应的残差结构。** 候选价值斜率的跨度 B 在预测发行时保存，之后用 B² 加权已完成残差。风险矩阵包含误差偏差与相关性。若完整当前情境已经固定 B，加权只是标量缩放；额外结构收益针对不同决策情境的混合，不能凭加权公式宣称普遍改进。
3. **从融合到可执行准入。** 求解器返回可行性和 KKT 残差；数值失败明确返回 fallback。协调器使用 `-lambda_uncertainty * U * Q_i`，其中 Q_i 随候选变化，因此会影响选择。全部源缺失或快照过期时禁止启动更新和探测。

## 最小融合调用

```python
import numpy as np
from da_rf_fusion import DecisionAwareFusion, FusionConfig, QualityObservation

fusion = DecisionAwareFusion(3, context_dim=2, config=FusionConfig())
quality = [QualityObservation(True, 50, 0, 0.02, diagnostic=0.2)
           for _ in range(3)]

snapshot0 = fusion.issue(
    window=0,
    predictions=np.array([0.20, 0.25, 0.30]),
    context=np.array([0.4, 0.8]),
    quality=quality,
    slopes=np.array([0.0, 0.5, 1.0]),
)
# 第 0 个预测对应第 1 个窗口的目标；只有完成后才允许送入标签。
fusion.observe_label(forecast_window=0, target=0.28, arrival_window=1)
snapshot1 = fusion.issue(
    1, np.array([0.30, 0.32, np.nan]), np.array([0.4, 0.8]),
    [quality[0], quality[1], QualityObservation(False, 0, 0, None)],
    np.array([0.0, 0.5, 1.0]),
)
print(snapshot1.weights, snapshot1.forecast, snapshot1.uncertainty)
print(fusion.get_last_matrices())
```

`predictions` 必须是同一未来目标的预测，不能直接传入三个含义不同的变化分数。质量只描述观测支持、年龄和统计波动。loss-only 服务模式使用绝对损失的支持与 bootstrap 方差，不能伪造缺失的相邻服务诊断。

在线使用 `DelayedRidgeForecaster.issue()` 时，它和 `fusion.issue()` 使用相同的预测标识。目标到达后，分别向两个模块传入原标识、标签与实际到达窗口。旧标签可乱序到达，但事件时间不可后退。重拟合后的预测不会覆盖历史预测。永久不可得标签必须调用 `discard_pending()` 显式处理；不会自动当作零标签。

## 对照模式

`FusionConfig(moment_mode=...)` 支持：

| 模式 | 优化器改变 |
|---|---|
| `decision_full` | 完整决策加权矩阵，主方法 |
| `decision_diagonal` | 仅使用其对角线 |
| `unweighted_full` | 同掩码、同 kernel 的未加权完整矩阵 |
| `unweighted_diagonal` | 未加权对角矩阵 |
| `trace_matched` | 将未加权矩阵缩放到决策加权矩阵的迹 |
| `no_risk` | 去掉风险项，保留质量、惯性与 cap |
| `original_c` | 同目标的 Original RF-C 标量 MSE 乘法与 capped KL；去掉联合风险和惯性 |

模式只改变优化器，使用相同的完整决策矩阵计算下游 U，以隔离权重机制。单独改变 U 定义应作为另一实验因素。`require_complete_enabled_vectors=True` 是档案复用对照：任何启用源缺失时，该记录不增加联合矩阵支持。

`original_c` 是明确的组件对照，不是未经执行版本核验的完整 Original RF 复现。原稿的目标和调度公式存在不同版本；真实端到端复现仍需原代码、配置和运行记录。

## 接入恢复协调器

每个 `Candidate` 表示一个更新与推理配置对。调用方提供同一完整时域的：

- `gross_baseline`：不更新的 gross 服务收益 W；
- `gain`：同推理协议下，更新与不更新的 gross 收益差 G，包含部署前资源竞争与部署后队列影响；
- `cost`：整个实际占用区间的直接训练费用 C，不重复扣除已进入 G 的服务损失；
- `error_allowance`：W+G 的误差余量；
- 总资源需求、步数、部署延迟、工作线程可用性与恢复校准支持。

`CoordinatorConfig.alpha_cost` 必须与外部评价 reward 的成本系数一致。`select()` 只给出启动指令；调用方执行训练、更新占用容量、记录实际费用，并在 worker 完成后部署模型。无可行不更新配置时返回服务延期状态。

经验余量下的更新标记为 `conditional`。只有外部同时误差覆盖依据成立，并为比较候选提供证书声明，才可返回 `certified`；设置布尔值本身不会创造统计保证。探测始终是 `probe_not_certified`，启动时预留计划费用，实际执行费用另行核对。同一个旧损失不能重复计为多个持续损失窗口。

## 演示结果的解释

在线融合演示验证预测发行、迟到标签、部分源缺失、PSD、权重和快照记录。梯度恢复诊断在独立的合成二分类关系变化任务中，真实计算 logistic 梯度、使用 worker 副本并延迟部署；它检查“该漂移可通过更新恢复”的前提。

恢复诊断的更新时刻由独立日程指定，不能解释为 DA-RF 触发器的闭环收益。DQN/PPO、真实 edge queue、真实数据和全部候选的恢复校准仍需接入并另行运行。固定权重阈值验证中的约 7.81% 损失下降也只属于其声明的同边际、同尺度构造。

详细结果以随附 JSON/CSV 为准，记录随机种子、配置、源码 hash、可用掩码、发行/标签时间、部署时间、梯度步数与费用；没有把未执行的实验写成测量结果。
