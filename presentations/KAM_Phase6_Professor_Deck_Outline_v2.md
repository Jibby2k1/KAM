# Sparse Separable Memory for Sequence Models — Technical-Intuition Deck

Professor-facing presentation outline
Kernel-Adaptive Memory (KAM), Phase VI
Revision 2 · July 28, 2026

## Design contract

- Nine slides, approximately 12–15 minutes.
- Every explanatory visual has a neighboring equation or mathematical condition.
- The visual and equation use the same labels, colors, and variables.
- Equations are short enough to discuss verbally; derivations remain out of the main deck.
- Blue denotes geometry/routing or the focal KAM method.
- Amber denotes algebra/adaptation or evidentiary caveats.
- Preliminary evidence remains explicitly distinct from confirmation.

---

## Slide 1 — Sparse Separable Memory for Sequence Models

Research question:

> Can model capacity grow with a large memory bank \(M\), while active expert computation grows only with the retrieved set \(K\), where \(K \ll M\)?

The claim is complementary: attention models context; KAM supplies selectively accessed local transformations.

---

## Slide 2 — The technical thesis: capacity scales with \(M\), active experts with \(K\)

Visual:

- A bank of \(M\) supports.
- Only \(K\) selected supports activate their experts for one token.

Technical anchor:

\[
\text{stored expert parameters}\propto M,
\qquad
\text{active expert FLOPs}\propto K,
\qquad K\ll M.
\]

Important qualification:

\[
\text{routing cost}=
\begin{cases}
O(Md_z), & \text{exact search}\\
\text{sublinear / factorized}, & \text{approximate or product-key search.}
\end{cases}
\]

Interpretation: sparse experts solve only the expert-computation side; scalable routing is a separate technical question.

---

## Slide 3 — Routing is kernel localization

Visual:

- Query \(z\) in a key space.
- Elliptical distance contours defined by \(D\).
- Top-\(K\) keys highlighted.
- Soft weights shown only among selected keys.

Technical anchor:

\[
s_i(z)
=
-\frac{1}{2\tau}(z-k_i)^\top D(z-k_i),
\]

\[
\mathcal I(z)=\operatorname{TopK}_i s_i(z),
\qquad
a_i(z)=
\frac{e^{s_i(z)}}{\sum_{j\in\mathcal I(z)}e^{s_j(z)}}.
\]

Interpretation:

- \(k_i\) determines where support \(i\) applies.
- \(D\) shapes the neighborhood.
- \(\tau\) controls routing sharpness.
- Top-\(K\) makes the active support set sparse.

---

## Slide 4 — Retrieval is a local model, not just a stored vector

Visual:

- Each selected key points to a local affine expert.
- Routing weights blend the selected expert outputs.
- A gate adds the result to the baseline block.

Technical anchor:

\[
g_i(u)=A_i u+b_i,
\qquad
m(u)=\sum_{i\in\mathcal I(z)}a_i(z)g_i(u),
\]

\[
h^+
=
h_{\text{base}}^+
+
\gamma(u)W_o m(u).
\]

Interpretation:

- Geometry chooses a neighborhood.
- Algebra supplies the local transformation.
- A zero-initialized \(\gamma\) starts KAM near the baseline and lets memory earn influence.

---

## Slide 5 — “Separable” means two parameter classes and two timescales

Visual:

- Blue geometry track: keys/query map.
- Amber algebra track: values/local experts.
- Alternating update arrows.
- Geometry freezes near \(0.8T\), while algebra and the backbone continue.

Technical anchor:

\[
\theta_A
\leftarrow
\theta_A-\eta_A\nabla_{\theta_A}\mathcal L,
\qquad
\theta_G
\leftarrow
\theta_G-\eta_G\nabla_{\theta_G}\mathcal L,
\qquad
\eta_G\ll\eta_A,
\]

with the registered lifecycle

\[
\eta_G(t)=0 \quad \text{for } t\ge0.8T,
\]

\[
\left\|K_T-K_{0.8T}\right\|_2\le10^{-10}.
\]

Interpretation: learn the map, stabilize it, freeze it, then tune against stable routing.

---

## Slide 6 — The methods under exploration correspond to different mathematical interventions

| Method | Intervention | Scientific question |
|---|---|---|
| T-KAM-F | \(k_i\leftarrow\) fixed/data center | Is useful geometry available without learning keys? |
| T-KAM-L | \(k_i=k_i^{(0)}+\Delta k_i\) | Does learned support placement help? |
| T-KAM-ALT | alternate \(\theta_A\) and \(\theta_G\) updates | Does timescale separation improve optimization? |
| Online | \(\theta_{A,t+1}=\theta_{A,t}-\eta_{\rm on}\nabla\ell_t\) | Can algebra adapt quickly after a shift? |
| Dual memory | \(m_t=g_t m_t^G+(1-g_t)m_t^E\) | Do persistent and episodic memory complement each other? |
| Product-key | \(k_{pq}=k_p^{(1)}\oplus k_q^{(2)}\) | Can routing scale without exact search? |

Evidence boundary: one intervention cannot validate another.

---

## Slide 7 — Confirmation is a paired test of a prespecified estimand

Design:

- 156 fixed rows.
- 30 TinyStories primary pairs.
- 24 Tiny Shakespeare replication pairs.
- 50M training tokens per language row.
- Separate control and lifecycle cohorts.

Technical anchor:

\[
\delta_s
=
\log
\left(
\frac{L_{\text{KAM-F},s}}
{L_{\text{T-WIDE},s}}
\right).
\]

Primary pass:

\[
p_{\text{sign-flip}}\le0.05
\quad\land\quad
\operatorname{CI}_{95\%,\,\text{upper}}(\bar\delta)
\le
\log(0.98).
\]

Promotion:

\[
\text{PrimaryPass}\land\text{ReplicationPass}.
\]

Interpretation: the interval must clear a scientifically meaningful 2% improvement, not merely zero.

---

## Slide 8 — Intermediate evidence: large estimated effect, insufficient information

Pilot visual:

- T-KAM-F validation loss: 2.0081.
- T-WIDE validation loss: 2.8833.
- Zero-based bars.

Technical anchor:

\[
\frac{2.0081}{2.8833}-1=-0.304.
\]

Therefore, the pilot point estimate is 30.4% lower.

Uncertainty:

- \(n=3\) paired seeds.
- Exact paired permutation \(p=0.25\).
- Holm-adjusted \(p=0.75\).
- This is a screening result, not a confirmatory conclusion.

Campaign status as of July 28, 2026 at 6:40 PM EDT:

- 43/156 complete.
- 4 running.
- 109 pending.
- 0 failures.
- Partial scientific effects remain uninspected.

---

## Slide 9 — The decision decomposes into performance and mechanism gates

Technical anchor:

\[
\text{AccelerateFixed}
=
\text{PrimaryPass}
\land
\text{ReplicationPass}.
\]

\[
\text{ClaimLearnedGeometry}
=
\text{AccelerateFixed}
\land
\text{LifecyclePass}
\land
\text{LearnedVsFixedPass}.
\]

Decision paths:

1. Fixed KAM passes: accelerate fixed/data-centered geometry.
2. Fixed passes but learned lifecycle fails: retain fixed KAM; reject learned-memory claims.
3. Primary or replication fails: reject a general quality advantage; investigate only separately supported niches.

Discussion questions:

- Is the 2% margin the right scientific threshold?
- Should the next scale-up optimize held-out quality, active compute, or adaptation speed?
- Which mechanism result would most change the research direction?
