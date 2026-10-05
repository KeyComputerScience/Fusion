import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
refs=[]
def add(key,typ,title,authors,year,venue,category,claim,relevance,url,doi=None,**fields):
    refs.append(dict(key=key,type=typ,title=title,authors=authors.split(' | '),year=year,venue=venue,category=category,
                     primary_claim_paraphrase=claim,manuscript_relevance=relevance,primary_url=url,doi=doi,
                     verified_on='2026-10-02',verification='Primary publisher/proceedings or author institutional source inspected',**fields))

add('bates1969','article','The combination of forecasts','J. M. Bates | C. W. J. Granger',1969,'Operational Research Quarterly','forecast combination',
    'Combining forecasts exploits information in their error variances and covariance.',
    'Historical basis: using cross-source error dependence is established and is not itself the proposed novelty.',
    'https://www.tandfonline.com/doi/abs/10.1057/jors.1969.103','10.1057/jors.1969.103',volume='20',number='4',pages='451--468',
    note='The 1969 journal name is Operational Research Quarterly; the current publisher indexes it under Journal of the Operational Research Society.')
add('granger1984','article','Improved methods of combining forecasts','Clive W. J. Granger | Ramu Ramanathan',1984,'Journal of Forecasting','forecast combination',
    'Regression forecast combination can include an intercept and relax conventional restrictions on weights.',
    'A dynamic-mean and bias-aware comparator is necessary before attributing gains to full joint risk.',
    'https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3980030207','10.1002/for.3980030207',volume='3',number='2',pages='197--204')
add('timmermann2006','incollection','Forecast combinations','Allan Timmermann',2006,'Handbook of Economic Forecasting','forecast combination',
    'Correlation, instability, asymmetric loss and weight-estimation error determine when forecast combinations help.',
    'Motivates explicit net-value evidence and regularization instead of an unconditional diversification claim.',
    'https://www.sciencedirect.com/science/chapter/handbook/pii/S1574070605010049','10.1016/S1574-0706(05)01004-9',volume='1',pages='135--196',publisher='Elsevier',editor='Graham Elliott and Clive W. J. Granger and Allan Timmermann')
add('rubin1981','article','The Bayesian bootstrap','Donald B. Rubin',1981,'The Annals of Statistics','Bayesian uncertainty',
    'The Bayesian bootstrap assigns random Dirichlet masses to observed support points.',
    'Our posterior concerns retained blocks of action errors, rather than posterior probabilities over RF models.',
    'https://people.eecs.berkeley.edu/~jordan/sail/readings/rubin.pdf','10.1214/aos/1176345338',volume='9',number='1',pages='130--134')
add('hoeting1999','article','Bayesian model averaging: A tutorial','Jennifer A. Hoeting | David Madigan | Adrian E. Raftery | Chris T. Volinsky',1999,'Statistical Science','Bayesian uncertainty',
    'Bayesian model averaging propagates uncertainty over competing models into predictions and decisions.',
    'Separates model-index uncertainty from the proposed posterior uncertainty of delayed fusion-error means.',
    'https://sites.stat.washington.edu/www/research/online/hoeting1999.pdf','10.1214/ss/1009212519',volume='14',number='4',pages='382--417',
    note='Author institutional PDF is the corrected version including discussion; original article text alone is cited as382--401 in the author publication list.')
add('yao2018','article','Using stacking to average Bayesian predictive distributions (with discussion)','Yuling Yao | Aki Vehtari | Daniel Simpson | Andrew Gelman',2018,'Bayesian Analysis','Bayesian uncertainty',
    'Stacking combines predictive distributions under proper scoring rules; Bayesian-bootstrap stabilization is among the compared approaches.',
    'Prediction combination and bootstrap weighting are prior art; our distinction is action-error covariance and realized deployment utility.',
    'https://sites.stat.columbia.edu/gelman/research/published/stacking.pdf','10.1214/17-BA1091',volume='13',number='3',pages='917--1007',
    metadata_primary_url='https://par.nsf.gov/biblio/10097765-using-stacking-average-bayesian-predictive-distributions-discussion')
add('bissiri2016','article','A general framework for updating belief distributions','P. G. Bissiri | C. C. Holmes | S. G. Walker',2016,'Journal of the Royal Statistical Society: Series B (Statistical Methodology)','Bayesian uncertainty',
    'General Bayesian updating can be formulated through losses rather than only a sampling-model likelihood.',
    'Frames decision-oriented inference while requiring transparent assumptions; it does not establish our posterior calibration.',
    'https://academic.oup.com/jrsssb/article/78/5/1103/7040623','10.1111/rssb.12158',volume='78',number='5',pages='1103--1130')
add('julier1997','inproceedings','A non-divergent estimation algorithm in the presence of unknown correlations','Simon J. Julier | Jeffrey K. Uhlmann',1997,'Proceedings of the 1997 American Control Conference','dependence-aware fusion',
    'Covariance intersection protects consistency when cross-correlations between estimation errors are unknown.',
    'Contrast conservative unknown-dependence fusion with our estimated joint action-error structure.',
    'https://sites.google.com/umsystem.edu/uhlmannj/home/publications','10.1109/ACC.1997.609105',volume='4',pages='2369--2373',publisher='IEEE',
    metadata_note='Authors/title/venue/year verified in coauthor institutional publication list; IEEE DOI page resolves but anti-robot page prevents extraction of pages. Pages cross-checked in Julier institutional papers, not independently from original paper body.')
add('uhlmann2003','article','Covariance consistency methods for fault-tolerant distributed data fusion','Jeffrey K. Uhlmann',2003,'Information Fusion','dependence-aware fusion',
    'Covariance union and intersection support consistent decentralized fusion despite incompatible or dependent estimates.',
    'Explains why independent-error treatment can double-count evidence; our posterior PSD estimate is not a CI consistency guarantee.',
    'https://www.sciencedirect.com/science/article/pii/S1566253503000368','10.1016/S1566-2535(03)00036-8',volume='4',number='3',pages='201--215')
add('denoeux2008','article','Conjunctive and disjunctive combination of belief functions induced by non-distinct bodies of evidence','Thierry Denoeux',2008,'Artificial Intelligence','dependence-aware fusion',
    'Cautious belief-function combination addresses overlapping, non-distinct evidence with idempotent fusion rules.',
    'Source dependence is a mature fusion problem; our actionable-error moments differ from belief-function combination.',
    'https://www.sciencedirect.com/science/article/pii/S0004370207001063','10.1016/j.artint.2007.05.008',volume='172',number='2--3',pages='234--264')
add('kittler1998','article','On combining classifiers','Josef Kittler | Mohamad Hatef | Robert P. W. Duin | Jiri Matas',1998,'IEEE Transactions on Pattern Analysis and Machine Intelligence','classifier fusion',
    'A common probabilistic framework analyzes combination rules for classifiers using distinct pattern representations.',
    'Provides late-fusion lineage; distinct representations do not imply independent action errors.',
    'https://dspace.cvut.cz/entities/publication/829206e1-4e28-4091-b0a7-14f2455efaff','10.1109/34.667881',volume='20',number='3',pages='226--239')
add('baltrusaitis2019','article','Multimodal machine learning: A survey and taxonomy','Tadas Baltrusaitis | Chaitanya Ahuja | Louis-Philippe Morency',2019,'IEEE Transactions on Pattern Analysis and Machine Intelligence','multimodal scope',
    'Multimodal learning encompasses representation, translation, alignment, fusion and co-learning.',
    'Defines our scope as source-forecast fusion with deployment decisions, not a new multimodal representation learner.',
    'https://arxiv.org/abs/1705.09406','10.1109/TPAMI.2018.2798607',volume='41',number='2',pages='423--443',
    metadata_note='Authors/title/content inspected in author arXiv source; final2019 issue/pages cross-checked in IEEE-indexed bibliographic record. Original DOI carries2018 online year.')
add('ma2022','inproceedings','Are multimodal transformers robust to missing modality?','Mengmeng Ma | Jian Ren | Long Zhao | Davide Testuggine | Xi Peng',2022,'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition','missing modalities',
    'Missing-modality robustness of multimodal transformers varies with dataset and fusion choices.',
    'Motivates forecast-mask evaluation; our source-output outage experiment does not establish robustness to physically missing candidate inputs.',
    'https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html',pages='18177--18186')
add('han2021','inproceedings','Trusted multi-view classification','Zongbo Han | Changqing Zhang | Huazhu Fu | Joey Tianyi Zhou',2021,'International Conference on Learning Representations','evidential fusion',
    'Trusted multi-view classification fuses Dirichlet class evidence while modeling uncertainty of individual views.',
    'Dirichlet class evidence differs from our block-mass posterior over action-error means.',
    'https://openreview.net/pdf?id=OOsR8BzCnl5')
add('sensoy2018','inproceedings','Evidential deep learning to quantify classification uncertainty','Murat Sensoy | Lance Kaplan | Melih Kandemir',2018,'Advances in Neural Information Processing Systems','evidential uncertainty',
    'Evidential classifiers parameterize uncertainty of class probabilities through a Dirichlet distribution.',
    'Avoid conflating evidential class uncertainty with posterior uncertainty of fusion-error moments.',
    'https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html',volume='31')
add('boyd2004','book','Convex optimization','Stephen Boyd | Lieven Vandenberghe',2004,'Convex Optimization','convex optimization',
    'Convexity, optimality and numerical solution principles support disciplined optimization formulations.',
    'Supports the PSD quadratic plus KL objective and explicit constrained feasibility; convexity itself is not novel.',
    'https://www.cambridge.org/highereducation/books/convex-optimization/17D2FAA54F641A2F62C7CCD01DFA97C4','10.1017/CBO9780511804441',publisher='Cambridge University Press')
add('amos2017','inproceedings','OptNet: Differentiable optimization as a layer in neural networks','Brandon Amos | J. Zico Kolter',2017,'Proceedings of the 34th International Conference on Machine Learning','optimization in learning',
    'OptNet incorporates optimization problems as differentiable neural-network layers.',
    'Places optimization-based fusion in prior art; our fixed-source convex decision rule is not an end-to-end differentiable layer.',
    'https://proceedings.mlr.press/v70/amos17a.html',volume='70',pages='136--145',series='Proceedings of Machine Learning Research')
add('donti2017','inproceedings','Task-based end-to-end model learning in stochastic optimization','Priya Donti | Brandon Amos | J. Zico Kolter',2017,'Advances in Neural Information Processing Systems','decision-focused learning',
    'Task-based learning fits probabilistic models through downstream stochastic-optimization performance.',
    'Decision utility is established; our contribution targets delayed multi-source fusion and cost-accounted deployment rather than RF training.',
    'https://proceedings.neurips.cc/paper_files/paper/2017/hash/3fc2c60b5782f641f76bcefc39fb2392-Abstract.html',volume='30')
add('wilder2019','inproceedings','Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization','Bryan Wilder | Bistra Dilkina | Milind Tambe',2019,'Proceedings of the AAAI Conference on Artificial Intelligence','decision-focused learning',
    'Decision-focused learning optimizes predictive models with respect to downstream combinatorial decisions.',
    'Distinguishes improved prediction from improved decisions; motivates measuring deployment actions and net gains.',
    'https://ojs.aaai.org/index.php/AAAI/article/view/3982','10.1609/aaai.v33i01.33011658',volume='33',number='1',pages='1658--1665')
add('spo2022','article','Smart "Predict, then Optimize"','Adam N. Elmachtoub | Paul Grigas',2022,'Management Science','decision-focused learning',
    'Smart predict-then-optimize measures prediction errors through downstream optimization loss and introduces a tractable convex surrogate.',
    'The new claim must concern retained joint action errors and execution-tested gating, not the first decision-aware prediction objective.',
    'https://pubsonline.informs.org/doi/pdf/10.1287/mnsc.2020.3922','10.1287/mnsc.2020.3922',volume='68',number='1',pages='9--26')
add('gibbs2021','inproceedings','Adaptive conformal inference under distribution shift','Isaac Gibbs | Emmanuel J. Candes',2021,'Advances in Neural Information Processing Systems','online calibration',
    'Adaptive conformal inference adjusts coverage behavior online in response to distribution shift.',
    'Basis for adaptive calibration; our delayed projected direct-quantile update has its own explicit accounting proof.',
    'https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html',volume='34')
add('barber2023','article','Conformal prediction beyond exchangeability','Rina Foygel Barber | Emmanuel J. Candes | Aaditya Ramdas | Ryan J. Tibshirani',2023,'The Annals of Statistics','nonexchangeable calibration',
    'Weighted and generalized conformal procedures quantify coverage degradation under nonexchangeability.',
    'Explains why drift and dependence invalidate automatic transfer of exchangeable conformal guarantees.',
    'https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.pdf','10.1214/23-AOS2276',volume='51',number='2',pages='816--845')
add('angelopoulos2024','inproceedings','Online conformal prediction with decaying step sizes','Anastasios Nikolas Angelopoulos | Rina Barber | Stephen Bates',2024,'Proceedings of the 41st International Conference on Machine Learning','online calibration',
    'Decaying step sizes balance retrospective coverage with stable quantile estimation in online conformal prediction.',
    'Useful recent comparator for calibration theory; our constant-step implementation must not claim this method is executed.',
    'https://proceedings.mlr.press/v235/angelopoulos24a.html',volume='235',pages='1616--1630',series='Proceedings of Machine Learning Research')
add('gibbs2024','article','Conformal inference for online prediction with arbitrary distribution shifts','Isaac Gibbs | Emmanuel J. Candes',2024,'Journal of Machine Learning Research','online calibration',
    'Adaptive step sizes achieve small online calibration regret over local time intervals under changing distributions.',
    'Positions constant-step calibration as a simpler implementation with narrower aggregate accounting guarantees.',
    'https://www.jmlr.org/papers/v25/22-1218.html',volume='25',number='162',pages='1--36')
add('zaffran2022','inproceedings','Adaptive conformal predictions for time series','Margaux Zaffran | Olivier Feron | Yannig Goude | Julie Josse | Aymeric Dieuleveut',2022,'Proceedings of the 39th International Conference on Machine Learning','time-series calibration',
    'Time-series calibration studies learning-rate effects and aggregates adaptive conformal procedures without a fixed rate choice.',
    'Motivates sensitivity analysis of history and calibration rates while distinguishing implemented constant-rate behavior.',
    'https://proceedings.mlr.press/v162/zaffran22a.html',volume='162',pages='25834--25866',series='Proceedings of Machine Learning Research')
add('angelopoulos2023','article','Conformal prediction: A gentle introduction','Anastasios N. Angelopoulos | Stephen Bates',2023,'Foundations and Trends in Machine Learning','uncertainty foundations',
    'Conformal prediction wraps predictive models with statistically justified uncertainty sets under its required calibration assumptions.',
    'Use to distinguish posterior credible uncertainty from empirically calibrated one-sided deployment bounds.',
    'https://www.nowpublishers.com/article/DownloadSummary/MAL-101','10.1561/2200000101',volume='16',number='4',pages='494--591',
    note='Use final journal title, not the longer arXiv manuscript title.')
add('gama2014','article','A survey on concept drift adaptation','Joao Gama | Indre Zliobaite | Albert Bifet | Mykola Pechenizkiy | Abdelhamid Bouchachia',2014,'ACM Computing Surveys','drift adaptation',
    'Concept-drift adaptation separates changing prediction relationships, detection strategies and evaluation methodology.',
    'A detected change alone does not show that a costly RF update can recover service value.',
    'https://doi.org/10.1145/2523813','10.1145/2523813',volume='46',number='4',pages='44:1--44:37',article_number='44')
add('bifet2007','inproceedings','Learning from time-changing data with adaptive windowing','Albert Bifet | Ricard Gavalda',2007,'Proceedings of the 2007 SIAM International Conference on Data Mining','drift detection',
    'Adaptive windowing adjusts memory to observed changes and controls false detections under stated conditions.',
    'Drift detection is not identical to recoverability or cost-beneficial model deployment.',
    'https://epubs.siam.org/doi/10.1137/1.9781611972771.42','10.1137/1.9781611972771.42',pages='443--448',publisher='SIAM')
add('joulani2013','inproceedings','Online learning under delayed feedback','Pooria Joulani | Andras Gyorgy | Csaba Szepesvari',2013,'Proceedings of the 30th International Conference on Machine Learning','delayed feedback',
    'Delayed-feedback reductions characterize how delayed observations alter online learning regret.',
    'Supports issue-time versus observation-time bookkeeping; it does not directly prove our deployment safety.',
    'https://proceedings.mlr.press/v28/joulani13.html',volume='28',number='3',pages='1453--1461',series='Proceedings of Machine Learning Research')
add('gneiting2007','article','Strictly proper scoring rules, prediction, and estimation','Tilmann Gneiting | Adrian E. Raftery',2007,'Journal of the American Statistical Association','forecast evaluation',
    'Proper scoring rules evaluate probabilistic forecasts while encouraging truthful predictive distributions.',
    'Calibration and probabilistic scores complement realized net service gain; none alone establishes useful deployment.',
    'https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf','10.1198/016214506000001437',volume='102',number='477',pages='359--378')
add('pdf2024','inproceedings','Predictive dynamic fusion','Bing Cao | Yinan Xia | Yi Ding | Changqing Zhang | Qinghua Hu',2024,'Proceedings of the 41st International Conference on Machine Learning','dynamic-quality fusion',
    'Predictive dynamic fusion learns collaborative beliefs and calibrates them using a generalization-oriented dynamic-fusion analysis.',
    'A close dynamic quality/confidence competitor; distinguish online delayed action-error archives and realized update costs.',
    'https://proceedings.mlr.press/v235/cao24c.html',volume='235',pages='5608--5628',series='Proceedings of Machine Learning Research')
add('qmf2023','inproceedings','Provable dynamic fusion for low-quality multimodal data','Qingyang Zhang | Haitao Wu | Changqing Zhang | Qinghua Hu | Huazhu Fu | Joey Tianyi Zhou | Xi Peng',2023,'Proceedings of the 40th International Conference on Machine Learning','dynamic-quality fusion',
    'Quality-aware dynamic fusion is analyzed through relations between fusion weights, uncertainty and generalization performance.',
    'Quality-dependent weighting is established; our narrow increment is posterior joint action-error uncertainty reused by weights and deployment.',
    'https://proceedings.mlr.press/v202/zhang23ar.html',volume='202',pages='41753--41769',series='Proceedings of Machine Learning Research')
add('deshpande2024','inproceedings','Online calibrated and conformal prediction improves Bayesian optimization','Shachi Deshpande | Charles Marx | Volodymyr Kuleshov',2024,'Proceedings of the 27th International Conference on Artificial Intelligence and Statistics','calibrated decision uncertainty',
    'Online calibration and conformal prediction improve uncertainty-guided Bayesian optimization in studied settings.',
    'Calibration plus downstream decisions is established; our deployment setting and closed-loop accounting are different.',
    'https://proceedings.mlr.press/v238/deshpande24a.html',volume='238',pages='1450--1458',series='Proceedings of Machine Learning Research')

for r in refs:
    r['url']=r['primary_url']
    r['author']=' and '.join(r['authors'])

out=dict(verified_on='2026-10-02',research_reference_count=len(refs),excludes_datasets=True,refs=refs,
    use_notes=[
    'Primary metadata includes publisher/proceedings and author institutional sources; explicit access limitations are retained in individual entries.',
    'Do not claim first context-adaptive, Bayesian, convex, task-aware or dependence-aware fusion; each has established prior art.',
    'The proposed Dirichlet posterior is over retained action-error block masses, not class probabilities or RF model indices.',
    'Current measured utilities use unit prices; heterogeneous decision importance is formulated but not empirically verified.',
    'Forecast-source outages do not test physical missing candidate-model input channels.',
    'Online calibration proof is aggregate delayed violation accounting; it is not conditional coverage for every accepted lease.',
    'The implementation uses fixed gamma=0.05; decaying or adaptively tuned steps are related work, not executed components.',
    'Five delay seeds per dataset are perturbation replicates, not five independent real sites or tasks.'
    ])
(ROOT/'foundational_refs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')

def tex(s):
    return str(s).replace('&',r'\&').replace('"',"''")

bib=[]
items=[]
for r in refs:
    fields={'title':r['title'],'author':r['author'],'year':r['year']}
    if r['type']=='article':fields['journal']=r['venue']
    if r['type'] in ('inproceedings','incollection'):fields['booktitle']=r['venue']
    for k in ('volume','number','pages','series','publisher','editor','doi','url'):
        if r.get(k):fields[k]=r[k]
    bib.append('@'+r['type']+'{'+r['key']+',\n'+',\n'.join('  '+k+' = {'+tex(v)+'}' for k,v in fields.items())+'\n}')
    # Plain bibitems are supplied for the current self-contained editor compiler.
    line=' and '.join(r['authors'])+", ``"+r['title']+",'' "+r['venue']
    if r.get('volume'):line+=', vol. '+r['volume']
    if r.get('number'):line+=', no. '+r['number']
    if r.get('pages'):line+=', pp. '+r['pages']
    line+=', '+str(r['year'])+'.'
    link=('https://doi.org/'+r['doi']) if r.get('doi') else r['url']
    items.append('\\bibitem{'+r['key']+'}\n'+tex(line)+'\n\\url{'+link+'}.')
(ROOT/'foundational_refs.bib').write_text('\n\n'.join(bib)+'\n')
(ROOT/'foundational_refs_bibitems.tex').write_text('% Primary bibliographic metadata checked 2026-10-02. Insert inside thebibliography.\n'+'\n\n'.join(items)+'\n')
print(json.dumps({'count':len(refs),'keys':[r['key'] for r in refs]},indent=2))
