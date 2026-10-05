# Performance安全性与可执行性审查（只读）

审查对象：当前`outputs/risk_rf_revision.tex`的Methodology、收益定理、Performance及执行审计。没有修改主文件或实验。当前版本已明确eta>=0、epsilon>=0；所述数学性质与冻结执行条件一致。

## 可以优势优先、且严格成立的核心结论

### 1. 先强调执行筛选确实包含成本与不确定性，而不是宣称统计安全已被保证

由于q>=0、s>0和epsilon>=0，任何获准动作均满足

    L=ĝ-5-q s>epsilon  =>  ĝ>5+epsilon+q s >=5+epsilon.

这意味着系统只为超过预设成本上界和不确定性惩罚的预测优势付费。它是准入的必要条件，是可检查的执行性质，不能直接推出真实D>0。

在固定同一w、h、mu、q、s和epsilon的当次状态下，后验gate的接受集合包含于对应point gate的接受集合。不能把这个局部包含关系直接扩大为所有独立调参的控制器、或整个会改变后续q的长时序之间的排序。

可直接使用的英文：

> Admission is an executable cost-and-uncertainty screen: every accepted lease has a corrected forecast exceeding the fixed interruption-and-fee charge, the issued uncertainty penalty and the admission margin. This is a necessary forecast condition, rather than an assumption that accepted updates are profitable.

### 2. 理论可以更明确地把“预测目标被覆盖”转化为“真实付费增益为正”

实际D=gtrue-lost_correct-3，保守目标Dtilde=gtrue-5，差额delta=2-lost_correct>=0。因此，对已获准且被覆盖的已发出预测，

    R<=q  =>  D>=Dtilde>=L>epsilon.

这比“存在会计恒等式”更有解释力：它说明gate选择的量与完整服务收益之间确实有可验证连接；安装、掉请求、恢复已包含在D中。条件R<=q必须明写。其真假要等完整标签到达后检查，不能表述成任意漂移下的事前安全承诺。

可直接使用的英文：

> Whenever an admitted lease's issued conservative target is covered, its complete net increment exceeds the admission margin. The nonnegative interruption slack transfers the lower-score condition to actual paid service return, including installation and restoration.

### 3. 优先展示真正非空的lower certificate，然后说明适用范围

定义

    C_T=sum a_j L_j - sum a_j s_j(R_j-q_j)_+.

定理逐条路径给出J_T-J_T^0>=C_T。C_T>0足以证明对应已完成轨迹的正参考相对收益；它是已完成反馈上的证书，不是准入前就可观测的保障。

HAR的报告值可以积极、精确地叙述为：五个固定延迟场景的平均C_T为+1.654，真实平均参考相对收益为+9.4。因为不等式在每个场景成立，平均界也成立；正平均证书证明这五个场景的平均收益为正。不要仅凭正平均证书宣称每个场景或新站点的证书均为正。三次HAR获准租期均覆盖且均受益，是另外独立的有限样本观察。

可直接使用的英文：

> The return bound is nonvacuous in the executed HAR replay: its mean weighted lower certificate is positive at1.654 utility units, beneath the measured9.4-unit mean improvement over the updating reference. The certificate is computed from matured issued forecasts and complete service outcomes; it verifies attained replay benefit under the lease contract.

MHEALTH的C_T=-25.6和真实差额-23.2也应保留，用一句说明界能如实反映失败，并不强制把失败解释成成功。占用任务a=0时C_T=0，表示没有新增租期暴露；不能称作正收益证书。

### 4. 先说实际减少有害动作，再区别对照与真实边界

- Room864：full拒绝全部五次point-gate有害部署；被拒租期净增量分别为-16,-38,-37,-19,-38。平均净效用高29.6，且保持更新reference的回报。相同B-gate也做到这一点，因此这是不确定性准入的执行收益，不是独有贝叶斯收益。
- MHEALTH：相对于joint point gate，有害租期均值从5.4降为2.2，即五个场景中27次减少到11次，描述性下降59.3%；净效用均值提高28.0。更具体的后验权重与gate交互避免了两个各-22的租期，full相对gate-only平均提高8.8。这些是执行效应，胜于只展示权重曲线。随后必须报告full仍比reference低23.2、比B-gate低6.6，以及12次获准租期中11次有害。
- HAR：full的三次获准租期均有正真实收益，平均相对joint高6.4、相对reference高9.4；在一项延迟场景中，以18的完整租期增益捕获B-gate错过的动作。相对B-gate平均3.6，但对应区间[-6.40,13.60]，不能宣称已建立稳定总体优势。
- Occupancy357：不发生获准动作，因而可以报告保持reference的实测行为，不能当作证明融合增益的任务。

可直接使用的英文：

> The controller changes costly actions, rather than merely their statistical representation. It avoids all five harmful point-gate deployments in Room864, reduces MHEALTH harmful admissions from5.4 to2.2 per replay, and executes three beneficial HAR leases without an observed harmful admission. The strongest aligned block-risk gate remains competitive, and MHEALTH still incurs a net loss relative to retaining the updating reference.

这里的“without an observed harmful admission”仅修饰HAR，不能扩展到四任务，也不能改成“safe deployment”。上述频率来自固定真实轨迹的延迟干预重复，不是独立群体风险估计。

### 5. 强调finite lease restoration带来的可控制、可重建暴露

H=4非重叠租期只在四窗口能完整落入计分区间时准入。candidate/reference在租期内固定，期末恢复共同reference。这样每个获准动作的模型暴露时长有界，并消除了未建模的永久延续差异；在单位效用、最多128个请求和规定收费下，D>=-133是保守物理损失界。

这提供的是暴露范围与完整收费合同，不是准确率安全、硬件SLA、事故损失或任意实际业务成本的界。共同准备与reference刷新工作在政策间相同；J-J0=sum aD的准确性依赖共享标签、共享模型准备和恢复的合同。

可直接使用的英文：

> Finite, nonoverlapping leases bound each installation's exposure and restore a common reference at closure. This execution contract makes the complete action increment reconstructible and prevents an uncharged continuation effect from being hidden inside a one-step forecast.

### 6. 会计一致性和防泄漏可以作为独立可执行性优势

可用证据：210条政策轨迹逐窗口独立重建部署模型、服务位置、正确预测、掉请求和收费；740个不同scenario fork、7665个method-specific checks；最大收费/服务差异2.28e-13。这不是仅把engine自己的local-net重新相加，因此能够支持实际执行与收益对象一致。

校准callback只在完整目标标签成熟后处理，并且永远使用issuance时保存的q计算violation。所有已发出fork，包括被拒候选，提供校准目标；这一机制在标签外生且两个固定模型可重建时，避免仅依赖获准子集反馈。它不需要真的部署所有候选。

未来truth字段不进入准入接口；guard测试修改离线未来truth不改变已发动作；另有empty/singleton与畸形输入测试、220个非空输入保持原执行。可以声称测试覆盖的接口不会通过这些future-truth字段作弊，不能声称已经证明整个软件生命周期不存在任何数据泄漏。

可直接使用的英文：

> The policy interface uses only issue-time information. Delayed callbacks respect label maturity and compare residuals with saved issued thresholds, including rejected candidate forks. Independent window-level service reconstruction confirms that the measured gains include the actual interruptions and fees rather than only a precomputed local objective.

## 不能从现有定理或数据推出的概括

1. “90%安全部署”“获准动作有90%保证正收益”“任意漂移下准入条件覆盖”。定理是all-issued的延迟regulator会计；MHEALTH admitted coverage0/12明确反例于这些经验概括。
2. “校准在所有任务成功”“不确定性始终可靠”“posterior credible scale就是未来租期收益的真实标准差”。Sigma与U是指定block law的不同矩；门控尺度是需要执行校准的特征。
3. “贝叶斯posterior全面优于非Bayesian baseline”。B-gate胜MHEALTH、占用任务平局、HAR只有一项额外动作且区间跨零。
4. “完整joint predictive covariance每个任务都必要”。joint与diagonal predictive的动作/回报在现有四任务相同；HAR/864选择lambda=0。
5. “权重改变即可带来收益”。Posterior-weights arm在四任务均无额外动作收益；有益证据来自门控穿越及其真实结果。
6. “提高eta/风险系数一定更安全/净收益更大”。eta单调减少w'Uw只在其他矩、参数固定时成立；h-mu、门控q及后续轨迹并不据此单调。风险敏感性是这些固定场景上的诊断结果。
7. “更多数据总让U下降”。只在所有block质量按比例增加、归一化分布不变时有确定收缩；新增不同block可以改变B、mu和Sigma。
8. “拒绝失败预测体现校准成功”。Room864所有informative fork都未覆盖，但被gate拒绝；拒绝损失与命中90%目标是两个评价维度。
9. “通过无conditional coverage假设的定理，保证所有流的正收益”。当前coarse count bound在短流上可松至无用；正lower certificate需要成熟后的实际residual。
10. “真实设备部署/SLA/恢复已验证”“physical missing-modality恢复有效”。实验是指定成本、延迟、预测通道outage的真实数据replay；没有物理输入丢失、硬件时延或长期生产恢复实测。
11. “五seed等于五独立站点”“跨参与者完全独立验证”。源轨迹重复，校准与test共享边界参与者；相邻HAR片段及lag有依赖。
12. “empty mask都实测发生且安全恢复”。empty mask留reference的行为来自production wrapper审计，不是四主任务的observed case。

## 推荐的优势优先顺序

1. 首段报告cost-inclusive utility与减少有害准入：三任务对joint mean提升、两类后验特定crossing。
2. 紧接用非空HARlower certificate、完整收费审计和泄漏/maturity guards解释这些收益为何与执行相连。
3. 保留最强B-gate与reference的比较，限定独有贝叶斯效应与MHEALTH失败；不用一开始先罗列所有限制。
4. 覆盖率段先呈现HAR执行覆盖、all-issued regulator一致性，再报告MHEALTH/Room864 informative失败，使覆盖与真实有害动作的区别清楚。
5. 敏感性突出移除eta恢复两个-22动作的可验证效应，然后报告零作用/负作用诊断；保持不选test敏感性赢家。

这种顺序可以增强读者对核心效应的理解，同时保留表格、分母、所有对照、失败结果与执行合同。不需要新增实验数字，也不应把严格会计性质包装为未经证明的普遍统计安全性。
