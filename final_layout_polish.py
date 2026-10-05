from pathlib import Path
root=Path('/Users/key/Documents/Codex/2026-10-02/jih')
for relative in ['outputs/risk_rf_performance.tex','outputs/risk_rf_revision.tex','work/performance_enhanced_body.tex']:
    p=root/relative
    t=p.read_text()
    for label in ['tab:data','tab:external-coverage']:
        mark=t.index('\\label{'+label+'}')
        start=t.index('\\begin{tabular}',mark)
        end=t.index('\\end{tabular}',start)+len('\\end{tabular}')
        if not t[max(mark,start-45):start].endswith('\\resizebox{\\linewidth}{!}{\n'):
            t=t[:start]+'\\resizebox{\\linewidth}{!}{\n'+t[start:end]+'}\n'+t[end:]
    t=t.replace(r' w_{i}^{\rm PDF}&=\operatorname{softmax}_s(r_{i,s}t_{i,s}),\qquad',r' w_{i}^{\rm PDF}&=\operatorname{softmax}_s(r_{i,s}t_{i,s}),\nonumber\\')
    t=t.replace(r' p_i^{\rm PDF}=\operatorname{softmax}_c',r' p_i^{\rm PDF}&=\operatorname{softmax}_c')
    p.write_text(t)
