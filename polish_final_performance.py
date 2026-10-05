from pathlib import Path
root=Path('/Users/key/Documents/Codex/2026-10-02/jih')
paths=[root/'outputs/risk_rf_performance.tex',root/'outputs/risk_rf_revision.tex',root/'work/performance_enhanced_body.tex']
for path in paths:
    txt=path.read_text()
    start=txt.index('\\section{Performance}')
    before,body=txt[:start],txt[start:]
    body=body.replace('Prefix & Audit & Calibration', 'Prefix & Error set & Calibration')
    body=body.replace('sigmoid-linear true-class-probability (TCP) heads on prefix audit records,', 'sigmoid-linear true-class-probability (TCP) heads on held-out prefix records,')
    body=body.replace('Uniform and singleton-source cases have explicit symmetric limits.', 'Zero-DU and singleton-source cases use explicit conventions.')
    body=body.replace('The ten-schedule column group is descriptive', 'The ten-schedule summary is descriptive')
    body=body.replace('Denominators show the available evidence, rather than masking sparse cases.', 'Exact denominators identify the available evidence in each stratum.')
    body=body.replace('Aggregate coverage can hide failures on decision-relevant\nepisodes.', 'Coverage of all issued and decision-relevant episodes.')
    body=body.replace('PDF-rule formula checks and request-level service reconstruction pass;', 'PDF and QMF formula checks and request-level service reconstruction pass;')
    path.write_text(before+body)
