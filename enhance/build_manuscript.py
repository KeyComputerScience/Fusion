"""Assemble a fully self-contained Performance section and native-editor document."""
from pathlib import Path
import subprocess,sys,json,re
ROOT=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(ROOT/'code/analyze_revision.py')],check=True)
s=(ROOT/'performance_template.tex').read_text()
for key,file in [('TABLE4','table4.tex'),('TABLE5','table5.tex'),('TABLE6','table6.tex'),('TABLE7','table7.tex'),('STATIONARY','stationary.tex'),('UPDATE_TABLE','update_diagnostic.tex')]:
 s=s.replace('@@'+key+'@@',(ROOT/'tables'/file).read_text())
sota=(ROOT/'sota_section.tex').read_text()
modern=json.loads((ROOT/'results/official_scalar/summary.json').read_text())
labels={'EF_CLASSIFIER':'Equal-logit classifier','QMF_SCALAR':'QMF core adaptation','PDF_SCALAR':'PDF core adaptation','REVISED_RISK_SCALAR':'Revised-risk classifier adaptation'}
for protocol in ['frozen','prequential_delayed_labels']:
 for method,label in labels.items():
  r=next(x for x in modern if x['method']==method and x['protocol']==protocol)
  cell=lambda field,d,scale:f'${r[field+"_mean"]*scale:.{d}f} \\pm {r[field+"_std"]*scale:.{d}f}$'
  line=label+' & '+cell('accuracy',2,100)+' & '+cell('nll',4,1)+' & '+cell('ece_10',4,1)+r' \\'
  pattern=r'^'+re.escape(label)+r' & .*? \\\\$'
  # Replace the next row in protocol order; the frozen row has already been consumed.
  marker='\n'+line+'\n'
  matches=list(re.finditer(pattern,sota,re.M))
  position=0 if protocol=='frozen' else 1
  match=matches[position]
  sota=sota[:match.start()]+line+sota[match.end():]
s=s.replace('@@SOTA@@',sota.replace(r'\begin{table*}[t]',r'\begin{table}[htbp]').replace(r'\end{table*}',r'\end{table}'))
u=(ROOT/'update_section.tex').read_text()
u=u.replace(r'Table~\ref{tab:update} reports this separate diagnostic',r'Table~\ref{tab:queue}C reports this separate diagnostic')
u=u.replace(r'Figure~\ref{fig:update_dose_response}',r'The released dose-response plots').replace('plots\nshows','plots\nshow')
s=s.replace('@@UPDATE_TEXT@@',u)
assert '@@' not in s
# Source location order originally placed ablations before queue: explicitly preserve table4/5/6/7 numbering.
# Table7 source is temporarily buffered until after table6 to keep counters and crossreferences correct.
a=s.index(r'\subsection{Component ablations');b=s.index(r'\subsection{Executed service')
abl=s[a:b];s=s[:a]+s[b:]
a=s.index(r'\subsection{Modern fusion');s=s[:a]+abl+s[a:]
# All tables are embedded, avoiding unsupported project-file dependencies in the native compiler.
s=s.replace(r'\centering\small',r'\centering\scriptsize')
s=s.replace(r'\begin{longtable}',r'\begingroup\scriptsize'+'\n'+r'\begin{longtable}')
s=s.replace(r'\end{longtable}',r'\end{longtable}'+'\n'+r'\endgroup')
(ROOT/'performance_section.tex').write_text(s)
preamble=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array}

\usepackage{url}

\setlength{\tabcolsep}{3pt}
\setlength{\emergencystretch}{2em}
\setcounter{section}{5}
\setcounter{table}{3}
\title{Quality-Conditioned Reliability Fusion for Adaptive Edge AI Services\\Performance Analysis: Executed Revision}
\author{}
\date{}
\begin{document}
\maketitle
'''
bib=r'''\begin{thebibliography}{9}
\bibitem{qmf} Q. Zhang et al. Provable Dynamic Fusion for Low-Quality Multimodal Data. Proceedings of ICML, PMLR 202, 2023. \url{https://proceedings.mlr.press/v202/zhang23ar.html}.
\bibitem{pdf} B. Cao et al. Predictive Dynamic Fusion. Proceedings of ICML, PMLR 235, 2024. \url{https://proceedings.mlr.press/v235/cao24c.html}.
\bibitem{boa} O. Wintenberger. Optimal learning with Bernstein Online Aggregation. Machine Learning, 2017. \url{https://arxiv.org/abs/1404.1356}.
\end{thebibliography}
\end{document}
'''
conclusion=(ROOT/'conclusion.tex').read_text() if (ROOT/'conclusion.tex').is_file() else ''
(ROOT/'performance.tex').write_text(preamble+s+conclusion+'\n'+bib)
print('Saved performance_section.tex (insertable) and performance.tex (standalone).')
