# Frozen specification

This file is the compact implementation contract. The detailed record and run
history are in `README.md` and `results/<profile>/RESULTS.md`.

1. Estimate `n_t = nbar_t + nhat_t` from Gustavo and Capital IQ only.
2. Condition exactly on total `n_t` at every Gustavo Q4 observation.
3. Use the historical mean quarterly allocation only in the slow-state drift.
4. Use both Capital IQ QoQ growth measures through noisy growth equations and
   leave unobserved quarters missing.
5. Parameterize state innovations by `tau` and `omega` and the cycle by a stable
   AR(2) damping/period representation.
6. Cut the state posterior from inflation.
7. On PPI × inverse markup × SPF QoQ data, estimate direct-only and free-static
   combined models, each without and with current/lagged real-oil controls.
8. Do not estimate lambda and do not impose an HSA restriction in this bundle.
9. Require measurement blocked prediction, convergence, exact-anchor, and free
   structural-learning gates before any later HSA restriction test.
