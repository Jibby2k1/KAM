# A retained-mass certificate for a query tile

A block can have a small mean key score and still hold a large fraction of softmax mass. A concentrated key can dominate the exponential sum. The following certificate uses the exact retained logits and upper bounds for the omitted groups, which are available from the keys and the current query tile.

## Derivation

Write $Z_S(q)=\sum_{i\in S}e^{q^\top k_i/\sqrt d}$ for the exact retained partition sum. Suppose $U_g(q)$ upper-bounds the same sum across each omitted group $g$. Then the dense probability assigned to $S$ satisfies

$$
m_S(q)\ \ge\ \frac{Z_S(q)}{Z_S(q)+\sum_{g\notin S}U_g(q)}.
$$

For a group of $n$ keys, let $s_i=q_0^\top k_i/\sqrt d$ be their scores at the tile representative and $\mu$ their mean. Suppose each $s_i$ lies in $[a,b]$. Convexity places $e^{s_i}$ below the chord through the interval endpoints, giving

$$
U_g(q_0)=n\left[\frac{b-\mu}{b-a}e^a+\frac{\mu-a}{b-a}e^b\right].
$$

The degenerate interval uses $n e^\mu$. Componentwise extrema give one valid interval. The group mean and greatest residual norm give another. Their intersection remains valid.

A second bound uses the score variance. Set $x_i=s_i-\mu$ and $B=b-\mu$. Taylor's integral remainder gives

$$
e^x=1+x+x^2\int_0^1(1-t)e^{tx}\,dt.
$$

The integral is increasing in $x$. For $x\le B$ it is bounded by $c(B)=(e^B-1-B)/B^2$, with $c(0)=1/2$. The linear terms average to zero. If $V$ bounds the variance of the centered scores, then $Z_g(q_0)\le ne^\mu(1+c(B)V)$.

For the centered key matrix $R$, a bound for $V$ is $\lambda_{\max}(RR^\top/n)\lVert q_0\rVert^2/d$. The maximum absolute row sum of the symmetric Gram matrix bounds the eigenvalue and is computed once per key group. Taking the minimum of the chord and variance bounds preserves validity.

For another query $q$ in the tile, the score increase is at most $\lVert q-q_0\rVert K_g/\sqrt d$, where $K_g=\max_i\lVert k_i\rVert$. Multiplying the representative upper bound by the exponential of this drift bound gives $U_g(q)$. This applies to whole earlier groups already visible to every query under test. A causal implementation can handle the current block at the exact per-query boundary.

## Controlled checks

For each regime, test twelve independently generated tiles of 32 queries at dimension 128, with 64 completed key blocks and four groups of 16 keys per block. Keep three mandatory blocks and choose five more by the existing maximum group-mean proxy on the representative query. Compute every dense logit in float64 for the verification. The compensated outlier adds a large score inside an excluded group and subtracts the mean shift from its other keys. The chosen shortlist stays the same.

| Key and query geometry | Actual median mass | Endpoint lower bound | Chord lower bound | Compact variance-assisted bound | Queries certified at 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coherent groups | 98.04% | 97.60% | 97.98% | 98.01% | 384 / 384 |
| Wide within-group residuals | 97.80% | 79.54% | 88.53% | 95.35% | 314 / 384 |
| Query drift in the tile | 97.89% | 96.28% | 96.86% | 96.91% | 384 / 384 |
| Compensated outlier | 38.15% | 3.57% | 31.44% | 22.71% | 0 / 384 |
| Independent keys | 14.00% | Below 0.001% | Below 0.001% | Below 0.001% | 0 / 384 |

The endpoint and chord rows use the intersected score interval. The compact variance-assisted variant uses only the mean, residual radius, Gram row-sum bound and key-norm bound. It certified every query in nine of the twelve wide-residual tiles at 95%. The version with componentwise extrema raised the compensated-outlier median bound to 31.44%, with selective fallback still warranted. Every tested lower bound stayed below the exact retained mass within the numerical test tolerance.

Exact membership is a stronger condition. A certificate can compare the minimum exact retained flexible-block mass with the largest upper bound among the omitted blocks. Only two of 1,920 queries met that strict condition. Several retained most of the mass while replacing a small block near the cutoff.

The compact variant adds three scalar summaries per group to the stored mean. With 128 BF16 mean coordinates and three FP32 summaries, storage grows from 256 to 268 bytes per group. Computing the small Gram bounds adds work during key summarization. The next evidence checkpoint is to measure certification rate and total runtime on real cached Llama query and key states, including conservative finite-precision bounds. The controlled check establishes the semantics and illustrates when the certificate can be too loose. [Output error and retained mass](ATTENTION_MASS_AND_RECONSTRUCTION.md)

[Replayable controlled results](../../results/general_benchmarks_layer_budget/shortlist-certificate.json)
