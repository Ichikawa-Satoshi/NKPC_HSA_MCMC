# Mixed-Frequency Competition Measurement (`annual_q4`)

**State-space treatment of annually observed competition data in the NKPC-HSA models.**

`annual_q4` is not a separate model. It is an observation scheme for the
competition series $N$: instead of pretending that PCHIP-interpolated quarterly
values are data, the model is told the truth — $N$ is observed once a year (in
Q4) and is missing in Q1–Q3. The Kalman filter / FFBS machinery handles the
missing observations exactly, with no approximation. This note writes the
scheme out in full.

---

## 1. Model overview

All HSA variants share the NKPC observation equation

$$
\pi_t = \alpha\,\pi_{t-1} + (1-\alpha)\,E_t\pi_{t+1} + \kappa_t\, x_t - \theta_t\, \hat N_t + e_t
$$

with the variant-specific slope restrictions

| Model | $\kappa_t$ | $\theta_t$ |
|---|---|---|
| `ces` | $\kappa$ (constant) | $0$ |
| `hsa_steady` | $\kappa_0 + \delta \bar N_t$ | $0$ |
| `hsa_dynamic` | $\kappa$ (constant) | $\theta$ (constant) |
| `hsa_const_theta` | $\kappa_0 + \delta \bar N_t$ | $\theta$ (constant) |
| `hsa_full` | $\kappa_0 + \delta \bar N_t$ | $\theta_0 + \gamma \bar N_t$ |

and the latent competition decomposition

$$
\hat N_t = \rho_1 \hat N_{t-1} + \rho_2 \hat N_{t-2} + u_t, \qquad
\bar N_t = n + \bar N_{t-1} + \varepsilon_t, \qquad
N^{obs}_t = \bar N_t + \hat N_t + \nu_t ,
$$

with $u_t \sim N(0,\sigma_u^2)$, $\varepsilon_t \sim N(0,\sigma_\varepsilon^2)$,
$\nu_t \sim N(0,\sigma_N^2)$, the AR(2) truncated to its stationary region.
Activity follows $x_t = \varphi_1 x_{t-1} + \zeta_t$, and the NKPC shock is
allowed to correlate with the activity shock through the control-function form
$e_t = \lambda \zeta_t + \eta_t$, $\eta_t \perp \zeta_t$ (`e_zeta_only`).

## 2. Observation timing set

The raw competition series is **annual** (one value per year, 31 observations
over 1982–2012). Define

$$
\mathcal{T}_N = \{\, t \in \{1,\dots,124\} : t \text{ is the fourth quarter of a year} \,\},
\qquad |\mathcal{T}_N| = 31 .
$$

Each annual value is placed at the Q4 quarter of its year as a stock
observation (`annual_timing: q4`) and transformed with the *same centering
constant as the quarterly-interpolated reference series*, so coefficients are
directly comparable across the two frequencies:

$$
N^{obs}_t = \frac{100 \log N^{annual}_{y(t)} - \bar c}{10},
\quad t \in \mathcal{T}_N, \qquad
\bar c = \overline{100 \log N^{quarterly\ ref}} .
$$

Under `quarterly_interpolated` (the default), the transformed annual data are
instead PCHIP-interpolated to all 124 quarters and every quarter carries an
$N$ measurement equation.

## 3. State-space form with time-varying observation dimension

Using `hsa_steady` as the concrete case, the state vector and transition are
(every quarter, both schemes):

$$
s_t = \begin{pmatrix} \hat N_t \\ \hat N_{t-1} \\ \bar N_t \end{pmatrix},
\qquad
s_t = \underbrace{\begin{pmatrix} 0 \\ 0 \\ n \end{pmatrix}}_{c}
+ \underbrace{\begin{pmatrix} \rho_1 & \rho_2 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}}_{F} s_{t-1}
+ w_t,
\qquad
w_t \sim N\!\big(0,\ \mathrm{diag}(\sigma_u^2,\, 0,\, \sigma_\varepsilon^2)\big).
$$

Define the adjusted inflation observation (the conditioning quantities are
known within the relevant Gibbs step):

$$
y^{\pi}_t \equiv \pi_t - \alpha\,\pi_{t-1} - (1-\alpha)\,E_t\pi_{t+1}
- \tfrac{\kappa_0}{100}\, x_t - \lambda\,\zeta_t
\;=\; \tfrac{\delta}{100}\, x_t \,\bar N_t + \eta_t .
$$

**Q4 quarters ($t \in \mathcal{T}_N$): two observation rows**

$$
y_t = \begin{pmatrix} N^{obs}_t \\ y^{\pi}_t \end{pmatrix}
= \underbrace{\begin{pmatrix} 1 & 0 & 1 \\ 0 & 0 & \tfrac{\delta}{100} x_t \end{pmatrix}}_{H_t} s_t + v_t,
\qquad
v_t \sim N\!\left(0,\ \begin{pmatrix} \sigma_N^2 & 0 \\ 0 & \sigma_\eta^2 \end{pmatrix}\right).
$$

**Q1–Q3 quarters ($t \notin \mathcal{T}_N$): one observation row**

$$
y_t = \big( y^{\pi}_t \big)
= \underbrace{\begin{pmatrix} 0 & 0 & \tfrac{\delta}{100} x_t \end{pmatrix}}_{H_t} s_t + v_t,
\qquad
v_t \sim N(0, \sigma_\eta^2).
$$

The only thing that changes across quarters is the dimension of
$(y_t, H_t, R_t)$. Equivalently, with the full observation vector
$y^{*}_t = (N^{obs}_t, y^{\pi}_t)'$ and a selection matrix $S_t$ ($I_2$ on Q4,
$(0\ 1)$ otherwise): $y_t = S_t y^{*}_t$, $H_t = S_t H^{*}_t$,
$R_t = S_t R^{*} S_t'$.

## 4. Likelihood: missing rows simply contribute nothing

$$
p(y_{1:T} \mid \Theta) = \prod_{t=1}^{124} p(y_t \mid y_{1:t-1}, \Theta),
\qquad
p(y_t \mid y_{1:t-1}) = N\!\big(y_t;\ H_t m_{t|t-1},\ H_t P_{t|t-1} H_t' + R_t\big).
$$

In Q1–Q3 the period contribution is a univariate normal (inflation only); in
Q4 it is bivariate. No imputation of $N$ appears anywhere — dropping the
missing row **is** the exact likelihood, not an approximation
(Harvey 1989, §3.4.7; Durbin & Koopman 2012, §4.10).

## 5. Filter recursions

Prediction runs every quarter:

$$
m_{t|t-1} = c + F\, m_{t-1|t-1}, \qquad P_{t|t-1} = F P_{t-1|t-1} F' + Q .
$$

The update uses whatever rows exist at $t$ (Joseph form; $K_t$ is $3\times 2$
in Q4 and $3\times 1$ otherwise):

$$
K_t = P_{t|t-1} H_t' \big( H_t P_{t|t-1} H_t' + R_t \big)^{-1},
$$
$$
m_{t|t} = m_{t|t-1} + K_t \big( y_t - H_t m_{t|t-1} \big),
\qquad
P_{t|t} = (I - K_t H_t) P_{t|t-1} (I - K_t H_t)' + K_t R_t K_t' .
$$

An important consequence: in Q1–Q3 the states are **not** blindly
extrapolated. The inflation row $H_t = (0,\ 0,\ \tfrac{\delta}{100}x_t)$ stays
active, so inflation itself updates $\bar N_t$ every quarter whenever
$\delta \neq 0$ (and updates $\hat N_t$ through $\theta$ in
`hsa_dynamic`/`hsa_const_theta`/`hsa_full`). State uncertainty collapses at
each Q4 anchor, grows between anchors, and collapses again — the model
performs its own interpolation, coherently with the transition dynamics and
with full uncertainty.

Initialization: $m_0 = (0,\ 0,\ N^{init}_0)'$, $P_0 = 10\, I_3$ (diffuse). The
linearly interpolated path $N^{init}$ is used **only** for starting values; it
does not enter the posterior.

## 6. FFBS: the backward pass needs no modification

Carter–Kohn backward sampling uses only the filtered moments:

$$
s_T \sim N(m_{T|T},\ P_{T|T}),
$$
$$
s_t \mid s_{t+1} \sim N\!\Big( m_{t|t} + A_t \big( s_{t+1} - m_{t+1|t} \big),\
P_{t|t} - A_t P_{t+1|t} A_t' \Big),
\qquad A_t = P_{t|t} F' P_{t+1|t}^{-1} .
$$

All missing-data information is already inside $(m_{t|t}, P_{t|t})$, so the
backward recursion is untouched. Within the Gibbs sampler this is data
augmentation (Tanner & Wong 1987): the Q1–Q3 values of
$(\hat N_t, \bar N_t)$ are latent draws refreshed every iteration, so their
posterior uncertainty propagates correctly into the coefficient posteriors —
the exact draw is from the joint posterior

$$
p\big(\Theta,\ \{\hat N_t, \bar N_t\}_{t=1}^{124} \ \big|\ \pi_{1:124},\
\{N^{obs}_t\}_{t \in \mathcal{T}_N}\big).
$$

## 7. Variance update uses observed quarters only

$$
\sigma_N^2 \mid \cdot \ \sim\
IG\!\left( a_N + \frac{|\mathcal{T}_N|}{2},\ \
b_N + \frac{1}{2} \sum_{t \in \mathcal{T}_N} \big( N^{obs}_t - \hat N_t - \bar N_t \big)^2 \right),
$$

i.e. 31 observations, not 124. This corrects the downward bias in
$\sigma_N$ that the interpolated scheme produced by counting 124
pseudo-observations.

## 8. `hsa_full` / `hsa_const_theta` two-block variant

These models sample $(\hat N, \bar N)$ with two exact conditional FFBS blocks.
The same principle applies within each block: e.g. in the
$\hat N \mid \bar N$ block the target row $N^{obs}_t - \bar N_t$ exists only
for $t \in \mathcal{T}_N$, while the inflation row is always present (and
symmetrically for $\bar N \mid \hat N$).

## 9. Why this matters over interpolation

| | PCHIP interpolation (`quarterly_interpolated`) | Mixed frequency (`annual_q4`) |
|---|---|---|
| Quarterly variation of $N$ | *Manufactured* by the spline (smooth pseudo-signal) | Generated by the AR(2)/RW transition, only as far as data allow |
| Interpolation uncertainty | Ignored (single imputation / generated-regressor problem) | Fully propagated (data augmentation) |
| $\sigma_N^2$ | Under-estimated (124 pseudo-obs) | Correct (31 obs) |

Empirically, this is what exposed the interpolation artifact: the apparent
"significance" of $\gamma$ (and partly $\theta$) at quarterly frequency
vanishes under `annual_q4` (posterior returns to the prior), while $\delta$ —
which loads on the trend $\bar N_t$, well pinned by the 31 annual anchors —
is essentially unchanged (unemployment-gap spec:
$\delta \approx +0.03$, Savage–Dickey $BF_{10} \approx 17\text{–}49$ at either
frequency).

## 10. Relation to the literature

The device is standard in mixed-frequency state-space econometrics:
Mariano & Murasawa (2003) for coincident indexes mixing monthly and quarterly
data; Aruoba, Diebold & Scotti (2009) for the ADS business-conditions index;
Schorfheide & Song (2015, *JBES*) for Bayesian mixed-frequency VARs with
exactly this missing-data-plus-simulation-smoother construction; and the
nowcasting literature generally (Giannone, Reichlin & Small 2008).

One modeling choice deserves note: the annual value is treated as a **Q4 stock
snapshot**. If one instead interpreted it as a within-year average, the Q4
observation row would become the aggregation constraint
$N^{obs}_{y} = \tfrac14 \sum_{q} (\bar N_q + \hat N_q) + \nu$, requiring
Mariano–Murasawa-style moving-average states. Since $N$ is a level-type
competition stock, the snapshot convention is the natural one.

## 11. Known limitation

The Chib marginal-likelihood routine
(`src/nkpc_hsa/gibbs/gibbs_marginal_likelihood.py`) does **not** implement the
missing-row branch of Section 4 and is fed the interpolated quarterly series
by the table pipeline; log-ML values for `annual_q4` blocks are therefore
invalid. Use the Savage–Dickey density ratios and posterior predictive scores
(both fully correct under `annual_q4`) for model comparison at annual
frequency.
