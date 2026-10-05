# Bayesian joint action-error fusion: executed closed-loop reproduction

本包对应同目录之外的 `risk_rf_revision.tex`，即已经在原 LaTeX 编辑器中修改并编译通过的完整 Methodology + Performance。源码内含三幅独立矢量图，不需要额外图片文件。HAR 已使用交付缓存完整复跑：全部九配置校准、选中参数、初始 quantile 及五个测试场景逐值完全一致，见 `portable_full_run_audit.json`。

## 实际完成了什么

这次加入的后验不确定性已经进入执行：`Σ + 4U` 决定凸优化权重，原始 `U` 决定校准 gate 的方向性尺度。`U = B/(A+1)` 表示所学动作误差修正的后验不确定性；`Σ` 是完整预测风险，二者不能混称为同一个方差。候选部署四个窗口后恢复共同参考模型，并核算服务丢失、部署费和恢复费。

四项真实多源任务的 full-minus-joint 平均净收益为 0、29.6、28.0、6.4。MHEALTH 中后验权重与 gate 联动，避免两个 -22 的部署；HAR 中后验浓缩接收了一次 +18 的有益部署，匹配非贝叶斯 B-gate 拒绝了它。这些是实际动作与收益证据，不是仅有权重变化的诊断。

结果也有明确边界：MHEALTH 的完整方法仍低于 B-gate 6.6、低于共享参考策略 23.2，信息性覆盖率仅 5/24。HAR 相对 B-gate 的 +3.6 仅来自一次额外动作，条件延迟区间跨零。864 上的收益是避免有害部署，而且与 B-gate 和 keep-reference 打平。不能将这些结果写成贝叶斯方法在所有域中独有且稳定的优势。源码保存全部结果及同信息、同 kernel、同调参预算对照。

## 环境与文件

实测环境为 Python 3.12.14、NumPy 2.3.5。核心算法、缓存读取和审计只需 NumPy；源码内的 TikZ/PGFPlots 直接生成图，科学绘图不是运行算法的依赖。可安装 `requirements.txt` 中的固定版本。

主要文件：

- `independent_bayes_fusion.py`：未修改的冻结基础实现。
- `independent_bayes_extension.py`：未修改的 MHEALTH/HAR 来源适配及预先声明的 diag-U gate 对照。
- `cached_runner.py`：仅替换穿戴数据读取的便携启动器，冻结算法文件保持字节不变。
- `production_bayes_guard.py`：显式处理零源弃权、单源权重 1、可行性约束；不读取未来 fork 标签。
- `new_bayes_fusion/`、`new_bayes_extension/`：协议、冻结标记、全部校准网格、选择配置、逐动作日志及审计结果。
- `new_bayes_sensitivity/`：固定选中控制器的质量、风险系数、后验惩罚、历史权重诊断；没有采用测试后“最优”配置。
- `independent_data/`：两项原始 occupancy CSV/TXT 与两项穿戴数据的无损算法输入缓存；来源及原始文件哈希随缓存保存。
- `analysis/analysis.json`：直接由逐动作日志重新计算的净收益、动作差异、覆盖计数和闭环下界。
- `prepare_raw_data.py`：可选的官方原始数据下载、压缩包校验和安全解压；用于从原始文件重建缓存。
- `manifest.json`：所有交付文件的 SHA-256。保留数据来源及署名信息，勿将构造参与者顺序称为真实全局时间。

## 一键复核现有结果

在本包根目录执行，缓存启动器会使用原始 occupancy 文件及完全一致的 wearable 输入数组：

```sh
python cached_runner.py audit_all_four_bayes.py
python cached_runner.py audit_canary_execution.py --results new_bayes_fusion --datasets occupancy357 occupancy864 --output verified_occupancy_service.json
python cached_runner.py audit_canary_execution.py --results new_bayes_extension --datasets mhealth319 har240 --output verified_wearable_service.json
python audit_production_bayes_guard.py
python analyze_posterior_trials.py --room-results new_bayes_fusion/results.json --wearable-results new_bayes_extension/results.json --out analysis_recomputed
```

上述审计重新拟合前缀模型、重建共同来源/候选/参考状态，并逐窗口核算服务，不使用保存的 `local_net` 来拼出服务收益。应复现 210 条服务轨迹、740 个不同场景 fork 和 7665 个方法 fork 检查；最大服务误差约 2.274e-13，校准恒等式误差约 6.484e-14，优化失败 0。

## 从头重跑同样的增强算法

先复制协议到新的输出目录，避免覆盖交付结果。命令以缓存数据运行，控制器源码和参数网格不变：

```sh
python -c "from pathlib import Path; import shutil; [(Path(d).mkdir(exist_ok=True), shutil.copy2(s,Path(d)/'protocol.json')) for d,s in [('rerun_room','new_bayes_fusion/protocol.json'),('rerun_wearable','new_bayes_extension/protocol.json')]]"
python independent_bayes_fusion.py --data independent_data --output rerun_room
python cached_runner.py independent_bayes_extension.py --data independent_data --output rerun_wearable --task mhealth319
python cached_runner.py independent_bayes_extension.py --data independent_data --output rerun_wearable --task har240
python analyze_posterior_trials.py --room-results rerun_room/results.json --wearable-results rerun_wearable/results.json --out analysis_rerun
```

复跑冻结标记中的运行时间会改变，部分输出哈希因此改变；控制器参数、输入、权重、动作和净收益应一致。不能将复跑标记的新时间当成原实验的预登记时间。缓存未包含原始 50Hz 的全部波形，而是包含算法实际使用的 float64 输入、类别、记录 ID 和参与者。`verify_cache_from_raw.py` 与原始来源哈希允许另行从官方原始数据检验。

固定选中配置的全部敏感性诊断：

```sh
python cached_runner.py diagnostic_bayes_sensitivity.py --output sensitivity_rerun
```

它使用保存的 prefix-only 配置和初始 quantile，并重新运行在线更新。质量指数 {0.5,1,2}、风险系数倍数 {0.5,1,2}、η {0,4,8}、β {0.90,0.97,1} 均逐项变化。输出包括动作差异、有害/有益部署和净收益，不能用它再选方法并声称主测试确认。

## 公平性、冻结与证据范围

两阶段协议分别在相应新数据读取前冻结。穿戴扩展是在 occupancy 结果之后声明的两项任务，因此不是四项任务在首次结果之前统一预注册。所有四项结果保留，没有继续搜索任务或按测试结果重构算法。此前 Air/Gas 探索结果不纳入此稿新主证据，也没有被改写为确认性结果；旧交付包独立保存。

每个主要对照采用同样九个 λ×margin 试验和三个 calibration-delay seeds，共 27 次配置场景评估；mean-only 的 λ 无效，名义九次实际含重复配置。模型工作、时间、反馈、kernel、同一联合档案、风险尺度和费用相同。B-gate 单独校准并有相同搜索预算；diag-U gate 在两个新穿戴任务上声明并执行；posterior-unused 必须逐动作等同 joint。

MHEALTH fit/calibration/test 参与者为 1–3、3–6、6–10；HAR 为 1–10、10–20、20–30。fit/test 不重叠，但 calibration/test 有边界参与者重叠。HAR 的原始片段有 50% 重叠，三阶协变量 lag 可跨参与者边界。五个延迟种子是同一真实轨迹上的队列干预，不是五个独立站点。所有 t4 区间仅描述这一条件下的延迟场景差异。

主实验 window=32，H=4，reference refresh=8；初始 500 步；每 4 窗口各模型训练 20 步。第一次 probe 用完整 prefix training buffer，在线标签首次进入后截为最新 512 条。参数 β=.97、kernel bandwidth=3、prior mass=2、archive=48、τ=.05、η=4、gate floor=1e-4、γ=.05、α=.1；部署费 2，恢复费 1，额外丢失 2 请求。第一次 source-outage 位于每 13 窗口中的前 2 个窗口，隐藏一个轮换的来源预测；物理候选输入没有丢失。

闭环定理要求非重叠、完整四窗口租约及租约内固定的参考/候选，恢复到共同参考，反馈外生，费用完整。它提供精确收益恒等式和条件下界，不保证所有漂移中更新都盈利。生产包装器验证边界行为，不包含持久化队列、重启恢复或现场硬件基准；这些落地要求已写明。

## 固定源文件哈希

基础实现：`15e3fc99d3f0935d2e66853b9b1949ef7825869c2f0d56a8ba49513838376e57`

扩展适配：`c70141db94d36b06079e89bd69e7a84c240938d2e40977cb807078cdb6f0253f`

第一阶段协议：`921eb4540ec9b6b9c159edd6b027c4620a6d5b04f13c5699a14089b64d35ac8d`

第二阶段协议：`98d0362f355f814e7207e36943f3e3b7ce724f9fd043b3e43153ed4b2ee0ec9c`

数据来源引用：UCI 357 DOI 10.24432/C5X01N；UCI 864 DOI 10.24432/C5P605；UCI 319 DOI 10.24432/C5TW22；UCI 240 DOI 10.24432/C54S4K。各数据的来源、处理及参与者限制同 LaTeX 文稿。
