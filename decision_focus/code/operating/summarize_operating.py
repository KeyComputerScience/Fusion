"""Postprocess the frozen all-point diagnostic; never reselect a policy."""
from pathlib import Path
import csv, gzip, hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
TASKS = ('rss348', 'arem366', 'gashome362')
LABEL = dict(precision_joint='Full', precision_diagonal='Diagonal penalty',
    scalar_mass='Scalar', no_posterior='No posterior', complete_only='Complete only',
    gls_exact='Exact GLS', block_sandwich='Block sandwich',
    factorized_information='Factorized information', pdf_mlp='PDF adaptation', qmf_mlp='QMF adaptation')
def load(p):
    if str(p).endswith('.gz'):
        with gzip.open(p, 'rt') as f: return json.load(f)
    return json.loads(Path(p).read_text())
def dump(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False))
def csvsave(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def slim(p):
    return {k:v for k,v in p.items() if k not in ('actions','seeds')}
def comparison(c, byid):
    f, q = byid[c['full_point']], byid[c['control_point']]
    return {k:v for k,v in c.items() if k!='cases'} | dict(
        control_arm=q['arm'], full_margin=f['extra_margin'], control_margin=q['extra_margin'],
        full_mean=f['mean_increment'], control_mean=q['mean_increment'],
        full_admissions=f['admissions'],control_admissions=q['admissions'],
        full_count_vector=f['admission_vector'],control_count_vector=q['admission_vector'],
        full_beneficial=f['beneficial'],control_beneficial=q['beneficial'],
        full_harmful=f['harmful'],control_harmful=q['harmful'],
        full_negative_loss=f['negative_loss'],control_negative_loss=q['negative_loss'])
def stats(matches, byid):
    def stat(xs):
        differences=[x['mean_full_minus_control'] for x in xs]
        return dict(pairs=len(xs),full_higher=sum(d>1e-10 for d in differences),
            equal=sum(abs(d)<=1e-10 for d in differences),full_lower=sum(d<-1e-10 for d in differences),
            difference_range=[min(differences),max(differences)] if differences else None,
            identical_actions=sum(x['changed_actions']==0 for x in xs),
            full_more_beneficial=sum(byid[x['full_point']]['beneficial']>byid[x['control_point']]['beneficial'] for x in xs),
            control_more_beneficial=sum(byid[x['full_point']]['beneficial']<byid[x['control_point']]['beneficial'] for x in xs),
            full_less_actual_loss=sum(byid[x['full_point']]['negative_loss']<byid[x['control_point']]['negative_loss'] for x in xs),
            control_less_actual_loss=sum(byid[x['full_point']]['negative_loss']>byid[x['control_point']]['negative_loss'] for x in xs))
    return dict(pooled=stat(matches),strict_count_vector=stat([x for x in matches if x['same_count_vector']]),
        by_arm={a:dict(pooled=stat([x for x in matches if byid[x['control_point']]['arm']==a]),
            strict_count_vector=stat([x for x in matches if byid[x['control_point']]['arm']==a and x['same_count_vector']])) for a in LABEL if a!='precision_joint'})
def row(p):
    return {k:p[k] for k in ('id','task','arm','extra_margin','fixed_margin','mean_increment','admissions','beneficial','harmful','zero','negative_loss','admitted_excess','admitted_max_excess','refusals')} | dict(
        admission_vector=json.dumps(p['admission_vector']),coverage_numerator=p['coverage'][0],coverage_denominator=p['coverage'][1])
def smallpoint(p):
    return {k:p[k] for k in ('id','arm','extra_margin','mean_increment','admissions','admission_vector',
        'beneficial','harmful','zero','negative_loss','admitted_excess','coverage')}
def smallcomparison(q):
    return {k:q[k] for k in ('full_point','control_point','control_arm','full_margin','control_margin',
        'same_pooled_count','same_count_vector','full_mean','control_mean','full_admissions','control_admissions',
        'full_count_vector','control_count_vector','full_beneficial','control_beneficial',
        'full_harmful','control_harmful','full_negative_loss','control_negative_loss',
        'full_excess','control_excess','mean_full_minus_control','categories','changed_actions',
        'shared_beneficial_count','shared_beneficial_units')}
def main():
    spec=importlib.util.spec_from_file_location('frozen_operating',OUT/'analyze_operating.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.verify()
    sources,_=m.saved_rows(); allpoints=load(OUT/'all_points.json.gz')
    compact=dict(protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        intervention='constant additional gate margin, immutable saved forecasts/callbacks, unchanged Fixed130/B130 ledger',
        units='net utility is a five-seed mean; action counts, actual negative loss and original-issued score excess are five-seed pooled sums',
        scope='Known-trace all-outcome development diagnostic; no test point selected for deployment or as a new primary policy.',
        comparison_warning='Matched admitted counts under an outcome-dependent permanent-loss ledger are descriptive; points and five delays are not independent replications.',
        score_excess_warning='Original-issued score excess is a forecast-undercoverage diagnostic, distinct from realized negative loss and from the deployment budget.',
        verification=load(OUT/'summary.json')['verification'], tasks={})
    flatmatches=[]; frontierrows=[]; fixedrows=[]; uniquerows=[]
    for t in TASKS:
        d=load(OUT/(t+'_diagnostic.json'));u=d['unique_points'];byid={p['id']:p for p in u}
        matches=load(OUT/(t+'_matched_comparisons.json.gz'))
        full=d['primary']['precision_joint']
        primary=[comparison(m.decompose(sources[t],full,p),byid) for a,p in d['primary'].items() if a!='precision_joint']
        pc=[comparison(x,byid) for x in d['primary_full_matched_count']]
        fs=[p for p in allpoints if p['task']==t and p['arm']=='precision_joint' and p['fixed_margin']]
        ext=[comparison(x,byid) for x in matches if byid[x['control_point']]['arm'] in ('pdf_mlp','qmf_mlp')]
        riskids=d['score_excess_frontier_ids'];lossids=d['realized_loss_frontier_ids']
        compact['tasks'][t]=dict(score_grid_size=d['score_grid_size'],raw_points=d['raw_points'],unique_action_patterns=d['unique_action_patterns'],
            primary={a:slim(p) for a,p in d['primary'].items()},primary_full_comparisons=primary,
            primary_full_same_pooled_count=pc,all_matched_summary=stats(matches,byid),
            all_external_same_count=ext,fixed_full=[slim(p) for p in fs],
            frontiers=dict(original_full_on_score_excess=d['original_full_on_score_excess_frontier'],
                original_full_on_actual_loss=d['original_full_on_realized_loss_frontier'],
                score_excess=[slim(byid[x]) for x in riskids],actual_loss=[slim(byid[x]) for x in lossids],
                score_only_ids=sorted(set(riskids)-set(lossids)),actual_loss_only_ids=sorted(set(lossids)-set(riskids))),
            all_nonzero_strict_comparisons=[comparison(x,byid) for x in matches if x['same_count_vector'] and abs(x['mean_full_minus_control'])>1e-10])
        uniquerows += [row(p) for p in u]
        fixedrows += [row(p) for p in allpoints if p['task']==t and p['fixed_margin']]
        frontierrows += [row(byid[x]) | dict(frontier=kind) for kind,ids in [('score_excess',riskids),('actual_negative_loss',lossids)] for x in ids]
        for x in matches:
            q=comparison(x,byid);flat={k:v for k,v in q.items() if k not in ('categories','full_count_vector','control_count_vector')}
            flat['full_count_vector']=json.dumps(q['full_count_vector']);flat['control_count_vector']=json.dumps(q['control_count_vector'])
            for cat,v in q['categories'].items():
                for unit,value in v.items():flat[cat+'_'+unit]=value
            flatmatches.append(flat)
    dump(OUT/'detailed_results.json',compact)
    lean={k:v for k,v in compact.items() if k!='tasks'} | dict(tasks={})
    for t,d in compact['tasks'].items():
        sums=d['all_matched_summary']
        byarm={a: {mode: {k:v for k,v in s.items() if k in ('pairs','full_higher','equal','full_lower','difference_range')}
            for mode,s in modes.items()} for a,modes in sums['by_arm'].items()}
        lean['tasks'][t]=dict(score_grid_size=d['score_grid_size'],raw_points=d['raw_points'],unique_action_patterns=d['unique_action_patterns'],
            primary={a:smallpoint(p) for a,p in d['primary'].items()},
            primary_full_comparisons=[smallcomparison(q) for q in d['primary_full_comparisons']],
            primary_full_same_pooled_count=[smallcomparison(q) for q in d['primary_full_same_pooled_count']],
            all_matched_summary={k:v for k,v in sums.items() if k!='by_arm'} | dict(by_arm=byarm),
            external_same_count_outcome_summary={mode: {a: sums['by_arm'][a][mode] for a in ('pdf_mlp','qmf_mlp')}
                for mode in ('pooled','strict_count_vector')},
            all_nonzero_strict_comparisons=[smallcomparison(q) for q in d['all_nonzero_strict_comparisons']],
            fixed_full=[smallpoint(p) for p in d['fixed_full']],
            frontiers={k:v for k,v in d['frontiers'].items() if k not in ('score_excess','actual_loss')}
                | dict(score_excess=[smallpoint(p) for p in d['frontiers']['score_excess']],
                    actual_loss=[smallpoint(p) for p in d['frontiers']['actual_loss']]))
    # Column-oriented tables avoid repeated long field names and duplicated primary records.
    pointcols=('arm','extra_margin','mean_increment','admissions','admission_vector','beneficial',
        'harmful','zero','negative_loss','admitted_excess','coverage')
    comparecols=('control_arm','full_margin','control_margin','full_mean','control_mean',
        'full_admissions','control_admissions','full_count_vector','control_count_vector',
        'same_pooled_count','same_count_vector','full_beneficial','control_beneficial',
        'full_harmful','control_harmful','full_negative_loss','control_negative_loss',
        'full_excess','control_excess','mean_full_minus_control','changed_actions',
        'shared_beneficial_count','shared_beneficial_units','categories')
    frontiercols=('id','arm','extra_margin','mean_increment','admissions','beneficial','harmful','negative_loss','admitted_excess')
    statcols=('pairs','full_higher','equal','full_lower','difference_range')
    tabular={k:v for k,v in lean.items() if k!='tasks'} | dict(
        schemas=dict(point_columns=pointcols,comparison_columns=comparecols,frontier_columns=frontiercols,
            by_arm_summary_columns=statcols,comparison_categories='counts and positive-magnitude pooled units; zero_return means changed zero-return actions'),
        full_detail_file='detailed_results.json',all_matched_csv='all_matched_comparisons.csv',
        all_fixed_margin_csv='fixed_margin_points.csv',all_frontier_csv='all_frontier_points.csv',tasks={})
    for t,d in compact['tasks'].items():
        sums=d['all_matched_summary']
        tabular['tasks'][t]=dict(score_grid_size=d['score_grid_size'],raw_points=d['raw_points'],unique_action_patterns=d['unique_action_patterns'],
            primary=[[p[k] for k in pointcols] for p in d['primary'].values()],
            full_vs_all_primary=[[q[k] for k in comparecols] for q in d['primary_full_comparisons']],
            full_primary_vs_all_same_positive_pooled_count=[[q[k] for k in comparecols] for q in d['primary_full_same_pooled_count']],
            all_matched_pooled=sums['pooled'],all_matched_strict_count_vector=sums['strict_count_vector'],
            matched_by_arm={a:{mode:[s[k] for k in statcols] for mode,s in modes.items()} for a,modes in sums['by_arm'].items()},
            external_actual_outcome_summary={mode:{a:sums['by_arm'][a][mode] for a in ('pdf_mlp','qmf_mlp')} for mode in ('pooled','strict_count_vector')},
            all_nonzero_strict_comparisons=[[q[k] for k in comparecols] for q in d['all_nonzero_strict_comparisons']],
            all_fixed_full=[[p[k] for k in pointcols] for p in d['fixed_full']],
            original_full_frontier_membership=dict(score_excess=d['frontiers']['original_full_on_score_excess'],actual_loss=d['frontiers']['original_full_on_actual_loss']),
            all_score_excess_frontier=[[p[k] for k in frontiercols] for p in d['frontiers']['score_excess']],
            all_actual_loss_frontier=[[p[k] for k in frontiercols] for p in d['frontiers']['actual_loss']])
    (OUT/'compact_results.json').write_text(json.dumps(tabular,separators=(',',':'),allow_nan=False))
    csvsave(OUT/'fixed_margin_points.csv',fixedrows);csvsave(OUT/'unique_points.csv',uniquerows)
    csvsave(OUT/'all_matched_comparisons.csv',flatmatches);csvsave(OUT/'all_frontier_points.csv',frontierrows)
    tex = r'''% Read-only experimental postprocessing fragment. No main manuscript was edited.
% All margins and all outcomes are preserved in the accompanying operating package.
\paragraph{Admission-count and operating-point diagnostics.}
We replay a declared additional margin grid
$\delta\in\{0,1,2,4,8,12,16,24,32,48,64,96,128\}$ on the
saved causal forecasts, retaining the original issued thresholds, all matured
callbacks, model versions and partial-history states. The stored paid gate
score includes the update fee, $G^{\mathrm{issued}}=F-qS-5$. The proposal is
$\mathbb{1}\{G^{\mathrm{issued}}>\delta\}$ and the same Fixed130/$B=130$ loss
ledger is then applied. A second, exhaustive breakpoint grid is the common union of positive
saved scores across methods in each task, rounded upward to $10^{-8}$ utility
units. It uses no outcome labels but does use future saved score states; it is
therefore a retrospective descriptive grid, not a prospectively selected
deployment rule. All points, exact action decompositions and both Pareto
frontiers are retained. Admitted-count matching also conditions on the
outcome-dependent ledger and is descriptive rather than a causal experiment.

At the unchanged primary point on RSS, Full and Scalar each admit ten leases.
Full has nine beneficial and no harmful leases, whereas Scalar has seven
beneficial and one harmful lease. Full's mean increment is $30.2$ versus
$14.8$: two additional beneficial leases contribute $76$ pooled utility and
one avoided harmful lease contributes $1$, giving $77/5=15.4$. Four actions
differ, including one zero-return action. The admission vectors differ,
$(2,4,2,1,1)$ for Full versus $(2,3,1,2,2)$ for Scalar; thus this finding
supports the primary pooled-count comparison, not a strict per-seed count match.

At Full's primary admitted count, every available strict five-seed count match
on RSS has exactly the same actions and return as Full (diagonal penalty,
no-posterior, exact GLS, block sandwich, and both neural adaptations). Full's
pooled original-issued scaled score excess is $3.7189$, versus $10.0519$,
$70.6554$, $3.7189$, $35.5968$, $127.6758$, and $140.4433$, respectively.
These are forecast-undercoverage comparisons on common admitted actions,
not additional net-utility gains. The exhaustive pooled-count match to
factorized information admits ten leases with mean increment $35.8$,
exceeding Full by $5.6$; its per-seed count vector differs and its score
excess is $9.7145$.

On AReM, all primary controls admit the same five harmful leases and tie at
$-30.0$. Full has scaled score excess $193.8715$, while Scalar and complete-only
are slightly lower at $188.8508$ and $193.6349$. On GasHome, all eight internal
controls admit the same five beneficial leases and tie at $45.2$ with zero
score excess. The primary neural adaptations admit eleven and ten beneficial
leases, obtain $52.8$ and $52.4$, and incur zero actual negative loss; after
matching Full's five-seed admission vector they retain Full's exact actions
and return, with score excess $55.3482$ and $65.3276$.

The score-excess and actual-loss frontiers answer different questions.
Full's original RSS point lies on neither exhaustive frontier: removing a
zero-return admission at a score breakpoint retains $30.2$ with lower score
excess, while other points attain larger return with zero actual negative
loss. AReM's rejection of every lease dominates all negative-return points.
GasHome Full lies on the score-excess frontier, shared with all seven internal
controls, but lies off the actual-loss frontier because the primary PDF
adaptation has higher return and the same zero negative loss. These diagnostics
do not establish an independent joint-matrix utility advantage on the fresh
tasks, universal lower score excess, or dominance in realized net return.

\begin{table}[t]
\centering\small
\caption{All declared additional-margin points for Full under the common
Fixed130/$B=130$ ledger. $J$ is the five-seed mean increment, $N$ the pooled
admitted count, and $E$ the pooled original-issued scaled score excess.
No margin is selected from these test outcomes.}
\label{tab:operating-full-fixed-diagnostic}
\begin{tabular}{r|rrr|rrr|rrr}
\hline
&\multicolumn{3}{c|}{RSS}&\multicolumn{3}{c|}{AReM}&\multicolumn{3}{c}{GasHome}\\
$\delta$&$J$&$N$&$E$&$J$&$N$&$E$&$J$&$N$&$E$\\\hline
'''
    for delta in m.FIXED:
        cols=[f'{delta:g}']
        for t in TASKS:
            p=next(p for p in compact['tasks'][t]['fixed_full'] if p['extra_margin']==delta)
            cols += [f"{p['mean_increment']:.1f}",str(p['admissions']),f"{p['admitted_excess']:.2f}"]
        tex += ' & '.join(cols)+r' \\'+'\n'
    tex += r'''\hline
\end{tabular}
\end{table}

Across all unique positive-count operating patterns, the pooled-count Full
comparisons on RSS are $19$ higher, $45$ equal and $29$ lower; the strict
five-seed count comparisons are $2$ higher, $45$ equal and $5$ lower. The
corresponding AReM pooled comparisons are $0/37/8$ and strict comparisons
$0/35/0$; GasHome pooled comparisons are $0/44/1$ and strict comparisons
$0/44/0$. These counts summarize a complete diagnostic set, not independent
replications or a test of statistical significance. Increasing the margin
can change the loss ledger and may increase subsequent admissions by avoiding
an earlier loss; admitted counts therefore need not be monotone for every
control. Independent service reconstruction over $15{,}950$ operating
trajectories and $600$ complete-lease forks gives zero service, fork and prefix
accounting discrepancies and no loss-budget violations. The future-outcome
perturbation checks cover the replay guard; forecast-state causality is
inherited from the saved experiment audits.
'''
    (OUT/'manuscript.tex').write_text(tex)
    m.verify()
    print('POSTPROCESS_PASS', 'compact_bytes', (OUT/'compact_results.json').stat().st_size)
if __name__=='__main__':main()
