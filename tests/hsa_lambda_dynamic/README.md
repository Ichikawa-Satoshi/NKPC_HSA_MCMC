# HSA NKPC with estimated lambda

This experiment estimates seven nested/paired specifications on one common cell:
PPI inflation, negative unemployment gap, SPF GDP-deflator expectations, and a
Gustavo annual effective-firm count allocated to quarters with Capital IQ.

The annual Gustavo change is allocated with the observed Capital IQ quarterly
profile when available. Cancellation-dominated annual ratios are continuously
shrunk toward a robust average quarterly profile. Missing Capital IQ years use
the robust average. Every Gustavo Q4 benchmark is reproduced exactly.

All specifications estimate the slow and cyclical competition states jointly
with the NKPC. The two HSA specifications estimate the multiplier `lambda`
rather than fixing it at six. The dynamic restriction is

```text
theta(N) = theta0 + gamma*N
kappa(N) = kappa0 + lambda*theta0*N + 0.5*lambda*gamma*N^2.
```

Run the smoke test first, then the production-length experiment and report:

```bash
python tests/hsa_lambda_dynamic/run.py --quick
python tests/hsa_lambda_dynamic/run.py
python tests/hsa_lambda_dynamic/build_report.py
```

Outputs are written under `tests/hsa_lambda_dynamic/results/`. The final PDF is
also copied to `output/pdf/hsa_lambda_dynamic_report.pdf`.

