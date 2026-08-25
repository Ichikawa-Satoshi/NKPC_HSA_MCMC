# Exact-N HSA decomposition

This experiment removes the competition measurement error from the state model:

```text
N_t = Nbar_t + Nhat_t                 (exactly)
Nbar_t = Nbar_{t-1} + eta_bar,t
Nhat_t = rho*Nhat_{t-1} + eta_hat,t
var(eta_bar) = omega*tau^2
var(eta_hat) = (1-omega)*tau^2.
```

Annual Gustavo changes are exact constraints. Quarterly allocation weights have
the robust average Capital IQ profile as their prior mean. In observed Capital IQ
years, the raw annual ratios update that prior with strength determined by the
within-year coherence statistic. Missing years retain the prior distribution.
Every sampled quarterly N path reproduces every Gustavo Q4 benchmark exactly.

The N posterior is estimated without inflation (a modular cut), then its draws are
integrated through each NKPC model. This propagates N uncertainty while preventing
weak Phillips-curve signals from rearranging the trend/cycle decomposition.

```bash
python tests/hsa_exact_n_decomposition/run.py --quick
python tests/hsa_exact_n_decomposition/run.py
python tests/hsa_exact_n_decomposition/build_report.py
```

