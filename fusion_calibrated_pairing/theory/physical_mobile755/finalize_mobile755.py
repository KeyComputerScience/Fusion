"""Outcome-complete reporting and immutable-callback audit; no policy edits."""
from pathlib import Path
import datetime, hashlib, json, sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_mobile755 as r


def main():
    freeze = r.verify()
    payload = r.load('mobile755/results.json.gz')
    selection = r.load('mobile755/selection.json')
    summary = r.load('mobile755/summary.json')
    common_rows = [row for g in payload['guarded'] if g['arm']=='paired'
                   for row in g['rows']]
    opportunities = dict(total=len(common_rows),
        beneficial=sum(row['local_net']>0 for row in common_rows),
        harmful=sum(row['local_net']<0 for row in common_rows),
        zero=sum(row['local_net']==0 for row in common_rows),
        potential_beneficial_gain=sum(max(0.,row['local_net']) for row in common_rows),
        potential_negative_loss=sum(max(0.,-row['local_net']) for row in common_rows),
        minimum=min(row['local_net'] for row in common_rows),
        maximum=max(row['local_net'] for row in common_rows),
        beneficial_origins=[dict(seed=g['seed'],origin=row['k'],D=row['local_net'],
                current_disagreement=row['disagreement'],information_ready=row['information_ready'],
                calibrated_score=row['gate_score'],empty_stratum=row['calibration_empty_stratum_fallback'])
            for g in payload['guarded'] if g['arm']=='paired'
            for row in g['rows'] if row['local_net']>0])
    index = {(x['arm'], x['recording'], x['seed']): x
             for x in payload['issued']}
    causality = []
    for raw in payload['raw']:
        key = raw['arm'], raw['recording'], raw['seed']
        issued = index[key]
        stream = dict(windows=400, decisions=[dict(
            k=row['k'], maturity=row['maturity'], N=row['N'],
            cal_context=row['cal_context']) for row in raw['rows']])
        conf = selection['selected'][raw['arm']]['config']
        pool = selection['score_pools'][raw['arm']]
        normal = r.cal.calibrated_run(stream, raw['rows'], conf, pool)
        for a, b in zip(normal['rows'], issued['rows']):
            assert a['k'] == b['k']
            assert a['gate_score'] == b['gate_score'] and a['action'] == b['action']
        for cutoff in (0, 76, 396):
            poisoned = r.cal.calibrated_run(
                stream, raw['rows'], conf, pool, poison_after=cutoff)
            compared = 0
            for a, b in zip(normal['rows'], poisoned['rows']):
                if a['k'] <= cutoff:
                    assert a['gate_score'] == b['gate_score']
                    assert a['action'] == b['action']
                    compared += 1
            causality.append(dict(arm=key[0], recording=key[1], seed=key[2],
                                  poison_after=cutoff, unchanged_issues=compared))
    record = dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Frozen mobile755 outcome-complete report; no method revision or test reselection',
        freeze_sha256=r.api.sha(ROOT/'mobile755_freeze.json'),
        source_count=freeze['source_count'], source_closure_verified=True,
        declaration_closure_verified=True,
        selection_sha256=r.api.sha(ROOT/'mobile755/selection.json'),
        selection_lock=r.load('mobile755/selection_lock.json'),
        policy_trajectories=len(payload['guarded']),
        budget_paths=len(payload['budget_rows']),
        method_specific_forks=sum(x['forks'] for x in summary['checks']['target_checks']),
        maximum_fork_error=max(x['maximum_error'] for x in summary['checks']['target_checks']),
        conservative_reported_fork_error_bound=8.89e-16,
        maximum_service_error=summary['checks']['max_service_error'],
        maximum_action_decomposition_error=max(
            x['accounting_error'] for x in summary['decomposition'].values()),
        callback_poison_trajectories=len(causality),
        callback_poison_unchanged_issues=sum(x['unchanged_issues'] for x in causality),
        callback_poison_checks=causality,
        actual_service_net_by_seed={arm:[sum(g['net'] for g in payload['guarded']
                    if g['arm']==arm and g['seed']==seed)
                    for seed in r.TEST_SEEDS] for arm in r.ARMS},
        state_fit_library_count=len(selection['prior_library']),
        score_prior_count={arm:len(pool) for arm,pool in selection['score_pools'].items()},
        primary_seed_count=5, primary_eligible_origins_per_seed=100,
        complete_issued_score_origins_per_arm=500,
        common_potential_opportunities=opportunities,
        all_outcomes_retained=True)
    r.dump('mobile755/execution_record.json', record)
    labels = dict(paired='Complete paired law',pair_factorized='Return--source factorization',
        full_gaussian='Full Gaussian moments',joint_diagonal_kernel='Joint KDE / diagonal kernel',
        unconditional='Unconditional joint residual',conditional_moment='Conditional mean/variance',
        pdf='Native-objective PDF adapter',qmf='Native-objective QMF adapter',dbf='Published DBF operator adapter')
    lines = [r'\subsection{Frozen mobile-phone collection validation}',
        r'\label{sec:mobile755-frozen}',
        'UCI755 is a separately collected two-modality mobile-phone dataset with ',
        'three accelerometer and three gyroscope coordinates. The official archive ',
        'was acquired only after the numerical source closure and physical protocol ',
        'were reviewed and separately authorized. A previous metadata Download ',
        'endpoint request returned an HTTP400 decoding error and exposed no sensor ',
        'or class values; this failed request is retained in the provenance record. ',
        'The successful transfer contains 31,991 native records. Original ordinal ',
        'order and both native target codes are retained. The CSV has no participant, ',
        'site or device identifiers, and its categorical timestamp does not establish ',
        'a validated physical clock. The evaluation therefore concerns one new ',
        'collection with an ordinal holdout. The two modalities are on a phone; ',
        'they are not asserted to be independent physical devices.',
        r'\begin{table}[t]\centering\small',
        r'\caption{Frozen mobile755 contiguous partitions. Native activity codes are encoded in sorted order; no class or row is removed.}',
        r'\label{tab:mobile755-partitions}',
        r'\begin{tabular}{lrrr}\toprule',
        r'Partition & Records & Native code 0 & Native code 1\\\midrule']
    for x in selection['metadata']['statistics']:
        a,b=x['native_class_counts']
        lines.append(f"{x['partition'].replace('_',' ')} & {x['records']:,} & {a:,} & {b:,} "+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}',
        'All nine pipelines receive the same nine calibration configurations and ',
        'three selection-delay schedules. Every selection configuration has zero ',
        'paid increment, so the declared tie rule selects bandwidth 1 and residual ',
        'probability 0.95 for every arm. All eight control co-winners are retained; ',
        'the predeclared arm order names the return--source factorization as the ',
        'locked primary control. The early partitions contain only native code 1 ',
        'and no candidate/reference disagreement sufficient to form a persistent ',
        'paired library. Its count is zero. The six information-law arms have no ',
        'eligible score prior, whereas each native fusion adapter has 24 completed ',
        'quiet prior scores. These facts are outcomes of the fixed partition rather ',
        'than criteria for replacing the task or changing the split.',
        r'\begin{table}[t]\centering\small',
        r'\caption{All nine locked mobile755 pipelines under budget 130. Returns are mean complete paid increments over five delay schedules; action and coverage counts pool those schedules descriptively. The last column is maximum optimistic excess on actual admissions.}',
        r'\label{tab:mobile755-complete}',
        r'\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrr}\toprule',
        r'Pipeline & Mean $\Delta J$ & Beneficial & Harmful & Issued coverage & Informative coverage & Admitted coverage & Max. excess\\\midrule']
    for arm in r.ARMS:
        x=summary['summary'][arm];c=x['coverage']
        cov=lambda s:f"{c[s]['covered']}/{c[s]['total']}" if c[s]['total'] else '--'
        lines.append(f"{labels[arm]} & {x['mean_increment']:.1f} & {x['beneficial']} & {x['harmful']} & {cov('issued')} & {cov('informative')} & {cov('admitted')} & {x['excess_max']:.3f} "+r'\\')
    lines += [r'\bottomrule\end{tabular}}\end{table}',
        'The paired controller makes no admission and matches every information-law ',
        'control. It avoids two losses made by each external adapter: $-9$ at origin ',
        '76 under seed 172001 and $-6$ at the same origin under seed 172002. Its ',
        'mean improvement over each adapter is three paid units. The paired score ',
        'at these origins is the physical-support fallback $-133$, while the PDF, ',
        'QMF and DBF scores are positive. This comparison records avoidance from ',
        'the complete frozen pipeline; because the factorized and moment controls ',
        'make the same decisions, it does not isolate an additional higher-order ',
        'joint-information gain. No beneficial deployment is observed in this ',
        'collection.',
        'The common complete potential outcomes contain six beneficial, 493 harmful ',
        'and one zero-return lease across 500 unique origin--delay pairs. Their ',
        'total potential positive gain is 77 units. Five positive leases occur at ',
        'origin 108 with gains 16, 14, 16, 15 and 15; their current-window ',
        'candidate/reference disagreement is zero although their full future ',
        'lease has positive return. The remaining positive lease is a one-unit ',
        'gain at origin 68 under seed 172002, before the paired information state ',
        'is ready. These measured opportunities are distinct from actual ',
        'admissions and from the restricted empty-stratum subset with a positive ',
        'ready raw score. That restricted subset contains zero events. No ',
        'information-law or external pipeline retains a positive lease here.',
        r'\begin{table}[t]\centering\small',
        r'\caption{Complete paired-minus-control action decomposition on mobile755. Each entry gives count/value across five delays; all eight controls are included. The mean difference divides the signed sum by five.}',
        r'\label{tab:mobile755-actions}',
        r'\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrr}\toprule',
        r'Control & Added gains & Avoided losses & Missed gains & Incurred losses & Mean difference\\\midrule']
    for arm,x in summary['decomposition'].items():
        t=x['terms'];fmt=lambda key:f"{t[key]['count']}/{t[key]['value']:.0f}"
        lines.append(f"{labels[arm]} & {fmt('added_gain')} & {fmt('avoided_loss')} & {fmt('missed_gain')} & {fmt('incurred_loss')} & {x['mean_difference']:.1f} "+r'\\')
    lines += [r'\bottomrule\end{tabular}}\end{table}',
        'There are 500 eligible score origins per arm and 19 informative origins ',
        'across the five schedules. Paired issued coverage is 489/500, ready-score ',
        'coverage is 269/280, and informative coverage is 14/19. Actual-admission ',
        'coverage is undefined because there is no admission. All three external ',
        'adapters have 10/19 informative and 0/2 admitted coverage. The maximum ',
        'admitted optimistic excess is 14.527 for PDF, 13.034 for QMF and 11.603 ',
        'for DBF. Their admitted effective calibration counts are one. These ',
        'denominators prevent interpreting aggregate coverage or abstention as ',
        'evidence of calibrated profitable admission.',
        'Under budget 130, every information-law arm has zero accumulated loss, ',
        'zero reserve and zero ledger refusal. Each external adapter incurs 15 ',
        'loss units across the five separately budgeted chains, with maximum ',
        'spent-plus-pending liability 130 in a chain. At budgets zero and 110 its ',
        'two proposals are refused and its return and loss become zero; at budget ',
        '260 its original two admissions and losses remain. Thus the common ',
        'reservation guarantee is exercised by the external proposals, while it ',
        'is inactive for the paired policy.',
        'All 45 primary trajectories, 180 budget paths and 4,500 method-specific ',
        'complete-fork checks pass independent request-level reconstruction. ',
        'Maximum service and four-term action-identity discrepancies are zero. ',
        r'Complete fork discrepancies are bounded by $8.89\times10^{-16}$. ',
        'Replaying stored immutable scores reproduces every issued score ',
        'and proposal exactly. In 135 future-complete-target poison replays, ',
        f"{record['callback_poison_unchanged_issues']:,} issuance comparisons before the poisoned maturity cutoff remain ",
        'unchanged. This audit holds forecasts fixed: source and service models ',
        'may consume individually arrived labels, whereas aggregate conditional ',
        'state and calibration callbacks require the whole target to mature. ',
        'The freeze, selection lock, original archive and complete result logs ',
        'are retained. The observed control ties and sparse admitted evidence ',
        'remain part of the reported collection.']
    (ROOT/'mobile755/performance_fragment.tex').write_text('\n'.join(lines)+'\n')
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.name not in ('manifest.json','finalize.log')]
    manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        numerical_source_closure_verified=True,freeze_sha256=record['freeze_sha256'],
        excluded_console_log='finalize.log is the live output stream of this reporting process; scientific result/audit files are included',
        source_hashes=freeze['algorithm_source_hashes'],
        retained_files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='callback_poison_checks'},indent=2))


if __name__=='__main__':
    main()
