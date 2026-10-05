"""Package measured primary analyses and the fully retained delay extension."""
from pathlib import Path
import csv,hashlib,json,shutil,zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs';WORK=ROOT/'work'
DEST=OUT/'performance_gain_evidence'
DEST.mkdir(exist_ok=True)
for name in ['protocol.json','pre_replay_freeze.json','original_files_before.json',
             'run_extension.py','portable_launcher.py','results.json','summary.json',
             'execution_audit.json','first_seed_benchmark.json','mechanism_cases.json',
             'portable_reproduction_audit.json']:
    path=WORK/'performance_delay_extension'/name
    if path.exists():shutil.copy2(path,DEST/name)
for name in ['performance_gain_analysis.json','performance_gain_review.md',
             'performance_safety_review.md','performance_revision_structure_audit.json',
             'performance_gain_reproduction_audit.json']:
    shutil.copy2(WORK/name,DEST/name)

src=(WORK/'performance_gain_analysis.py').read_text()
old="BASE = Path('outputs/bayes_closed_loop_repro')\nWORK = Path('work')"
assert old in src
src=src.replace(old,"""import argparse
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--package',type=Path,default=HERE.parent/'bayes_closed_loop_repro')
parser.add_argument('--output',type=Path,default=HERE/'analysis_rerun')
args=parser.parse_args()
BASE=args.package.resolve();WORK=args.output.resolve();WORK.mkdir(parents=True,exist_ok=True)""")
src=src.replace("'output_files':['work/performance_gain_analysis.json','work/performance_gain_review.md']",
                "'output_files':[str(WORK/'performance_gain_analysis.json'),str(WORK/'performance_gain_review.md')]")
(DEST/'primary_gain_analysis.py').write_text(src)

primary={}
for folder in ['new_bayes_fusion','new_bayes_extension']:
    primary.update(json.loads((OUT/'bayes_closed_loop_repro'/folder/'results.json').read_text()))
extension=json.loads((DEST/'results.json').read_text())
rows=[]
for phase,data,controls in [('primary',primary,['joint','bayes_gate','frequentist_gate','periodic','frozen']),
                            ('additional_delays',extension,['joint','bayes_gate','frequentist_gate'])]:
    for task,study in data.items():
        for trial in study['trials']:
            full=trial['results']['bayes_both']
            for mode in controls:
                base=trial['results'][mode]
                row=dict(phase=phase,task=task,seed=trial['seed'],comparator=mode,
                         full_net=full['net'],comparator_net=base['net'],net_difference=full['net']-base['net'],
                         served_correctness_change=full['gross']-base['gross'],saved_fees=base['fees']-full['fees'],
                         full_admissions=full['deployments'],comparator_admissions=base['deployments'],
                         full_harmful=full['harmful'],comparator_harmful=base['harmful'],
                         full_beneficial=full['beneficial'],comparator_beneficial=base['beneficial'])
                assert abs(row['net_difference']-row['served_correctness_change']-row['saved_fees'])<1e-8
                rows.append(row)
with (DEST/'all_paired_results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

readme='''# Performance gains and supplementary validation

本包配套更新后的 `risk_rf_revision.tex` 和 `risk_rf_performance.tex`。正文按实测增益、动作机制、完整收益分解和可核查执行保护重新组织；六个原主实验表格正文及所有原始结果保持不变。

新增实测：原四个数据轨迹、锁定控制器、72006–72010 延迟种子；4任务×5种子×4对照=80条新政策轨迹。Joint、posterior gate、full posterior、matched block gate 保持已选参数与初始 q。在线延迟校准继续执行。协议在新种子运行前写入，属于既有结果之后声明的诊断扩展；不作为新站点验证，不合并进原五种子主均值。

新 full-minus-joint 均值依任务为 0、36.2、32.6、5.0。MHEALTH full-minus-gate-only 为17.6；HAR full-minus-block 为6.8。最强对照与参考、负向种子、未覆盖预测和错失有益动作均保存在日志及正文。不能把减少有害动作等同于无条件安全或每次更新盈利。

## 复现

环境：Python3.12.14，NumPy2.3.5；依赖见旁边冻结 `bayes_closed_loop_repro/requirements.txt`。完整ZIP保留该67文件基础包，基础算法与数据读取适配文件完全不改。

从解压目录运行（可用 `--package` 指定基础包绝对路径）：

```sh
python performance_gain_evidence/primary_gain_analysis.py --package bayes_closed_loop_repro --output primary_analysis_rerun
python performance_gain_evidence/portable_launcher.py --package bayes_closed_loop_repro --output delay_validation_rerun
```

第二条会从实际输入重建模型、执行所有新延迟轨迹并独立逐请求核算。历史protocol/freeze原样保留；新执行时间写入 portable_launch_manifest，不把复跑时间当作原协议冻结时间。输出中的weights、actions、net和coverage应一致；elapsed/runtime/hash-of-launcher metadata可变化。

`run_extension.py` 是实际运行的冻结版本，`portable_launcher.py` 仅调整包和输出路径，不改变算法、协议或调参。`results.json` 包含所有新逐动作、校准callback和独立审计；`summary.json` 为汇总；`execution_audit.json` 核对80条trajectory与2960个method-specific fork，最大服务误差3.56e−14、零solver失败。`all_paired_results.csv` 包含原主试验与新增试验的全部配对收益、费用分解和动作数量，两个phase严格分列。

`performance_gain_analysis.json` 是原主试验新增核算：harmful43→11、beneficial保留4/8、服务与收费分解和完整动作收益分解。比例是固定轨迹重放的描述，不是跨站点风险估计。覆盖率分母随论文表格提供；后验矩阵与会计定理不替代实测校准。

`manifest.json` 保存交付文件hash。原算法、adapter和数据hash见基础包与新protocol。新文稿保持自包含TikZ/PGFPlots与内嵌文献，可在原LaTeX编辑器独立编译。
'''
(DEST/'README.md').write_text(readme)
manifest=dict(kind='retained primary gain analysis and actual fixed-controller additional delay experiments',
              primary_tables_preserved=6,additional_trajectories=80,additional_fork_checks=2960,
              primary_and_additional_averages_separate=True,source_selection='No task, seed, outcome or sensitivity winner removed',
              outputs_sha256={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='manifest.json'},
              manuscript_sha256=hashlib.sha256((OUT/'risk_rf_revision.tex').read_bytes()).hexdigest(),
              performance_source_sha256=hashlib.sha256((OUT/'risk_rf_performance.tex').read_bytes()).hexdigest())
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(OUT/'Performance_Gain_Validation_Repro.zip','w',zipfile.ZIP_DEFLATED) as z:
    for folder in [DEST,OUT/'bayes_closed_loop_repro']:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT))
    for name in ['risk_rf_revision.tex','risk_rf_performance.tex']:
        z.write(OUT/name,name)
print('Packaged',len(rows),'retained paired rows and all additional trajectories.')
