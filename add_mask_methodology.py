from pathlib import Path
p=Path('/Users/key/Documents/Codex/2026-10-02/jih/outputs/risk_rf_revision.tex')
s=p.read_text()
def replace_once(old,new):
    global s
    assert s.count(old)==1, (old[:100],s.count(old))
    s=s.replace(old,new)
replace_once('its quality-constrained estimation and influence rule, and the explicit bridge',
             'its availability-conditioned PSD archive and quality-constrained influence rule,\nand the explicit bridge')
replace_once('When a complete stored forecast vector and its common label arrive, define',
             'For the previous Risk RF control, when a complete stored forecast vector and\nits common label arrive, define')
replace_once('The unweighted matrix is retained as a diagnostic and as the Risk RF\ncontrol. The proposed optimizer instead uses the decision-weighted matrix\ndefined next.',
             'This complete-enabled-source update specifies the historical Risk RF control.\nThe new method and its matched controls reuse jointly observed subsets through\nthe archive below and retain a context-matched unweighted moment.')
a=s.index('The completed forecast archive')
b=s.index('The estimation target',a)
s=s[:a]+r'''Every archive record stores its original forecast mask $\mathcal A_r$, issued
context, slope span, predictions, and subsequent target-arrival time. The
completed archive retains both full and partially observed vectors whose
targets have arrived. For the current forecast-valid set $\mathcal A$, define
\begin{equation}
 \mathcal I_k(\mathcal A)=
 \{r:\text{target arrived before issuance }k,\ \mathcal A\subseteq\mathcal A_r\}.
 \label{eq:mask-archive}
\end{equation}
Only coordinates actually observed together in a saved record are reused.
For $r\in\mathcal I_k(\mathcal A)$, set
\begin{equation}
 a_{k,r}=\beta^{k-r}
 \exp\left(-\frac{\norm{\boldsymbol z_k-\boldsymbol z_r}_2^2}
                 {2h_z^2}\right),\qquad 0<\beta\le1,\ h_z>0.
\end{equation}
Bandwidth, archive length, initial matrices, and decay are fixed on historical
validation data. With $a_0>0$ and $M_0\succeq0$,
\begin{equation}
 \widehat M_k^{\mathcal A}=
 \frac{a_0M_{0,\mathcal A\mathcal A}+
       \sum_{r\in\mathcal I_k(\mathcal A)}a_{k,r}\chi_r
       \boldsymbol e_{r,\mathcal A}\boldsymbol e_{r,\mathcal A}^{\mathsf T}}
      {a_0+\sum_{r\in\mathcal I_k(\mathcal A)}a_{k,r}},
 \quad \boldsymbol e_{r,\mathcal A}=
 \boldsymbol p_{r,\mathcal A}-y_{r+1}\boldsymbol1.
 \label{eq:context-moment}
\end{equation}
For an unweighted matched-context control, replace $\chi_r$ by one and
$M_0$ by a declared PSD residual prior $R_0$ in the same expression, obtaining
$\widehat R_k^{\mathcal A}$. Its kernel, eligible records, and normalization
are identical. Nonnegative outer-product weights preserve PSD. The prior is a
fallback, not imaginary observations. No unavailable coordinate is zero-filled,
and no independently normalized pairwise deletion is used.

This reuse rule permits valid joint errors from the observed sources to update
their matrix while another enabled source is unavailable. Returning to a larger
mask requires records genuinely covering that larger set. For
$\mathcal A\subseteq\mathcal B$,
$\mathcal I_k(\mathcal B)\subseteq\mathcal I_k(\mathcal A)$: smaller masks
can reuse at least the same eligible records, without manufacturing information
about missing coordinates. Record exact-mask and superset-mask counts, context
support, prior fraction, and the age of the latest eligible residual.
Kernel weights neither create independent samples nor correct informative
missingness. The default containing-mask rule pools eligible observation strata.
An exact-mask treatment uses only $\mathcal A_r=\mathcal A$; it sacrifices
support to reduce mask-distribution mismatch and does not inherit the nested-set
property. If outage depends on unobserved loss, neither treatment identifies an
unconditional moment without an additional observation model.

'''+s[b:]
replace_once(r'S_k=\widehat M_k+\delta_k I',r'S_k=\widehat M_k^{\mathcal A_k}+\delta_k I')
a=s.index('At each window boundary, learn from a rolling archive')
b=s.index('A narrower context kernel',a)
s=s[:a]+r'''At each window boundary, learn from completed decision cases and freeze the
weights throughout the following window. The empirical criterion is
\begin{equation}
 \widehat{\mathcal L}_k(\boldsymbol w)=
 \boldsymbol w_{\mathcal A_k}^{\mathsf T}
 \widehat M_k^{\mathcal A_k}\boldsymbol w_{\mathcal A_k}.
\end{equation}
Unless a mask superscript is explicit below, $\widehat M_k$ denotes this active
matrix; matrix products in the optimizer and theory use active coordinates.
We call it a local pooled archive moment. '''+s[b:]
replace_once(r'r_k^{\rm pred}=\boldsymbol w_k^{\mathsf T}R_k\boldsymbol w_k.',
             r'r_k^{\rm pred}=\boldsymbol w_{k,\mathcal A_k}^{\mathsf T}'+
             '\n'+r' \widehat R_k^{\mathcal A_k}\boldsymbol w_{k,\mathcal A_k}.')
replace_once(r'f_{0,k}=\frac{a_0}{a_0+\sum_r a_{k,r}}',
             r'f_{0,k}=\frac{a_0}{a_0+\sum_{r\in\mathcal I_k(\mathcal A_k)} a_{k,r}}')
replace_once('where $N_k^{\\rm joint}$ counts genuinely observed archive vectors, and',
             'where $N_k^{\\rm joint}=|\\mathcal I_k(\\mathcal A_k)|$ counts eligible\njointly observed archive vectors, and')
replace_once('genuine residual observation and is not reset by kernel reweighting or prior',
             'eligible residual observation (dated by forecast origin, not late arrival)\nand is not reset by kernel reweighting or prior')
p.write_text(s)
print('Updated current methodology with availability-conditioned joint archive.')
