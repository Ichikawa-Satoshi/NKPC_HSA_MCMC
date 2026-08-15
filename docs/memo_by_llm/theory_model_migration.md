# Theory-model migration: fixed and moving local references

Current theory revision: `2026-08-moving-reference-hsa-v1`

The immutable pre-migration map is in
`docs/repository_production_call_chain_before_migration.md`. This document is the
post-migration map and implementation/provenance handoff.

## 1. Independent derivation from Fujiwara--Matsuyama Eq. 21--23

At a fixed symmetric steady state, Eq. 21 gives the quarterly local Rotemberg
Phillips curve coefficients

```
kappa0 = (zeta0 - 1) / chi
Theta0 = (1 / chi) * (1 - rho0) / rho0
mu0    = zeta0 / (zeta0 - 1).
```

Eq. 22 and symmetry, `s(z)=1/N`, imply
`d log N / d log z = zeta - 1`. From Eq. 23 and
`mu=zeta/(zeta-1)`,

```
d log mu / d log z = -(d zeta / d log z) / [zeta(zeta-1)]
(1-rho)/rho        =  (d zeta / d log z) / [zeta(zeta-1)].
```

Consequently

```
d zeta / d log N  = zeta * (1-rho)/rho
d kappa / d log N = zeta * Theta.
```

Under the Second law, `d zeta/dz>0`; since `d log N/d log z=zeta-1>0`,
both the slope response and the entry coefficient are positive:
`theta0>0` and `kappa_N>0`. This maintained law is distinct from the
domain conditions `chi>0`, `zeta0>1`, `mu0>1`, and `kappa0>0`.

The empirical competition state is
`Nbar=10(log N-log N0)` and quarterly inflation is stored in percentage
points. Hence `theta0=10*Theta0`,
`d_kappa_d_logN=10*kappa_N_empirical`, and

```
100*kappa_N_empirical = b_x*zeta0*theta0.
```

`KAPPA_SCALE=100` does not add another economic conversion. The sampler uses
`x/100`, so its internal competition-slope coefficient is exactly
`b_x*zeta0*theta0`; storage divides it by 100. Unit tests pin all three forms.

The equality is only a quarterly structural-coefficient restriction. The
current direct four-quarter YoY regression is rejected for R1--R3 rather than
silently receiving this equality. A future coefficient-preserving YoY design
must aggregate four quarterly structural observation equations.

## 2. Restriction taxonomy

| Category | Content | Code / metadata key |
|---|---|---|
| Theory identities / cross-equation relations | Fixed-reference equations above; R1--R3 impose only `100*kappa_N=b_x*zeta0*theta0` | `restriction_taxonomy.theory_identities`, `exact_restrictions` |
| Maintained HSA laws | Second law in F0/R1/R2/R3; weak Third law only in R2 | `maintained_laws` |
| Domain / admissibility | `zeta0>1`, `mu0>1`, `kappa0>0`; optional full-path `kappa_t,theta_t>0` | `domain_admissibility_conditions`, diagnostic rejection rates |
| Empirical approximation | Slow-moving `Nstar_t`, linear `kappa_t` and `theta_t` paths | `moving_reference_approximation` |
| Deliberately not imposed | Structural level restrictions determining `kappa0` and `theta0` from `chi,rho0,zeta0` | `restriction_taxonomy.deliberately_not_imposed`; reserved for future R4 |

Path failures are persisted as admissibility diagnostics. A draw share of at
least five percent triggers the explicit interpretation that the local linear
approximation may span too wide an observed competition range.

## 3. Model namespace, restrictions, and nesting

| Slug | Level | Reference | Sampled / derived coefficients | Maintained laws | Sampler |
|---|---|---|---|---|---|
| `hsa_f0` | F0 | fixed `N0` | constant `kappa0`; `theta0`; no `Nbar` | Second | fixed-reference 2-state FFBS |
| `hsa_u` | U | moving `Nstar_t` | `kappa_N`, `theta0`, `gamma` independent | neither | Particle Gibbs |
| `hsa_r1` | R1 | moving | `theta0,gamma` sampled; `kappa_N=b_x*zeta0*theta0/100` derived | Second | Particle Gibbs |
| `hsa_r2` | R2 | moving | R1 plus `gamma<=0` | Second + weak Third | Particle Gibbs |
| `hsa_r3` | R3 | moving | R1 plus `gamma=0` | Second; constant-pass-through benchmark only | exact joint FFBS |

Within one data/observation/prior family, `U superset R1 superset R2`. R3 is
`R1 + gamma=0`. F0 is not in this nesting relation: it fixes the reference and
has no `Nbar` state, whereas U/R1/R2/R3 estimate a slow-moving reference.

## 4. Historical slug mapping and reproducibility

| Historical slug | New report position | Preservation rule |
|---|---|---|
| `ces` | historical CES benchmark | slug, facade, sampler, run directory unchanged |
| `hsa_steady` | varying-slope reduced-form ablation | legacy `delta` retained |
| `hsa_dynamic` | entry-channel reduced-form ablation | legacy state/covariance definition retained |
| `hsa_const_theta` | constant-theta reduced-form ablation | not renamed R3 |
| `hsa_full` | unrestricted reduced-form full ablation | not renamed U |

`configs/models.yaml::models`, the legacy dispatch behavior, the legacy
`ESTIMATION_REVISION`, and `results/runs` remain intact. New runs use
`results/theory_runs/2026-08-moving-reference-hsa-v1`. The original
`nkpc_hsa_report.tex/.pdf` remains a separate historical report and never
relabels old `delta` draws. The new hierarchy has its own
`nkpc_hsa_restriction_report.tex/.pdf`.

## 5. Post-migration production call chain

```
raw price indexes / expectations / activity / competition
  -> scripts/01_build_data.py
  -> model_ready.csv with Q/Q and legacy 4Q-YoY inflation columns
  -> configs/models.yaml::theory_data_specs
  -> scripts/10_estimate_theory_models.py
  -> inference.wrappers.run_model
       -> theory_models.py registry + design validation
       -> hsa_theory fixed-reference or moving-reference dispatch
       -> Gibbs coefficient blocks + fixed FFBS / joint FFBS / Particle Gibbs
  -> results/theory_runs/<theory revision>/<cell>
       posterior.nc + metadata.json + priors.json + data_spec.json
  -> scripts/11_run_theory_diagnostics.py
       convergence + whole-path admissibility diagnostics
  -> scripts/19_build_theory_report.py
       exact signature validation + comparability validation
  -> results/tables/theory/*
  -> scripts/build_restriction_report.py
  -> report/nkpc_hsa_restriction_report.tex / .pdf
```

The legacy `scripts/02_estimate_models.py` call chain continues separately.

## 6. Data, inflation, expectations, and competition changes

- Price-index loaders now produce non-annualized Q/Q percentage-point columns
  alongside the unchanged four-quarter transformations.
- `Epi_qoq=Epi/4` is explicitly recorded as a quarterly-rate equivalent of a
  one-year source, not mislabelled as a next-quarter forecast. The unverified
  information-date convention is retained in metadata rather than guessed.
- The production competition transform remains
  `(100*log(N)-sample mean)/10`; its zero corresponds to the geometric-mean
  sample anchor `N0`, which is saved numerically.
- F0 uses deviations around that fixed anchor and contains no moving trend.
- Moving models keep the annual-Q4 versus interpolated competition observation
  machinery. Q/Q and 4Q-YoY inflation are mutually exclusive run-level designs;
  no likelihood loads both series.
- Restricted gap-proxy runs require an explicit `marginal_cost_loading=b_x`.
  The driver never assumes `b_x=1` for unemployment or output gaps.

## 7. Sampler and posterior-schema changes

The new dispatch entries live alongside, not over, the historical entries.
R1/R2 use Particle Gibbs because `gamma*Nbar*Nhat` is bilinear. R3 uses exact
joint FFBS after `gamma=0`. F0 uses a separate fixed-reference two-state FFBS.

New stored names are `kappa_N_empirical`, `d_kappa_d_logN`, `zeta0`, and `mu0`.
R1/R2/R3 do not store legacy `delta`; `kappa_N_empirical` is recomputed for each
draw. Paths remain `kappa_t` and `theta_t`. F0 stores `N_deviation`, while moving
models store `Nbar` and `Nhat`. U stores `kappa_N_empirical` as independently
sampled. Particle-Gibbs mixing statistics and admissibility-rejection statistics
remain visible.

## 8. Diagnostics and convergence

The existing R-hat/ESS pipeline now recognizes the new scalar and path names.
`scripts/11_run_theory_diagnostics.py` writes diagnostics in a separate revision
namespace and finalizes `convergence_status` in run metadata. The convergence
rule remains max R-hat 1.01 and minimum bulk/tail ESS 400; nonstationary AR(2)
draws and nonfinite posteriors are errors. Whole-path positivity is reported
separately as local-model admissibility, not called an HSA law.

## 9. Report section migration

| Historical report role | Restriction report role |
|---|---|
| Eq. 21 followed directly by the old time-varying regression | fixed-reference theory -> F0 -> explicit moving-reference approximation |
| steady/dynamic/const-theta/full as main hierarchy | historical reduced-form / ablation appendix |
| unrestricted old `delta` interpreted as theory slope response | never promoted; new `kappa_N_empirical` only |
| one generic theoretical-restriction label | four-way taxonomy table plus deliberately-not-imposed row |
| `nkpc_hsa_report` scans `results/runs` | `nkpc_hsa_restriction_report` scans only `results/theory_runs/<revision>` |
| revision-only stale filtering | content signature plus exact definition/restriction/data comparability checks |

The restriction report's primary order is theory, F0, moving-reference
assumption, U, R1, R2, R3, observation-design comparison, and robustness. Until
production MCMC runs exist, the preview says `CURRENT THEORY ESTIMATES PENDING`;
the production build fails. `build_restriction_report.py --allow-missing-runs`
creates only that explicit scaffold and never substitutes a historical number.
The original report and builder are unchanged.

## 10. Changed files and principal functions

| File | Principal change |
|---|---|
| `theory_models.py` | unique registry, taxonomy, unit mappings, activity/observation guards |
| `provenance.py` | content signatures and stale-artifact exception |
| `gibbs/hsa_theory/model.py` | F0/U/R samplers and new output names |
| `gibbs/hsa_full_pg/model.py` | opt-in cross-restricted coefficient block, R2 inequality, R3 FFBS, admissibility statistics |
| `inference/wrappers.py` | new dispatch, separate output namespace, complete metadata/provenance |
| `inference/diagnostics.py` | new variables and local-range diagnostic |
| `dataprep/func_data_build.py` | Q/Q inflation and quarterly-rate expectation equivalent |
| `config.py`, `configs/models.yaml` | separate theory data/model grids |
| scripts `10`, `11`, `19` | estimate, diagnose, and report the theory hierarchy |
| `reporting/theory_report.py` | exact current-run loader and comparability guard |
| `build_restriction_report.py`, restriction report `.tex` | separate theory-only PDF; historical report untouched |
| `run_restriction_production.py` | clean-revision estimate -> diagnostics -> validated artifacts -> PDF |

For U/R1/R2/R3, coefficient admissibility is sampled with a coordinate-Gibbs
kernel for the exact linearly truncated Gaussian conditional. This replaces an
independent rejection loop that can fail when whole-path positivity has low
conditional probability. One unconstrained probe per iteration is retained
only as an admissibility-pressure diagnostic; it does not enter the posterior.
The annual-Q4 production configuration gives the slow moving-reference models
400,000-iteration, four-chain runs based on the failed 12,000- and
120,000-iteration pilot diagnostics, while F0 retains its shorter converged
configuration. Independent model/spec runs are
executed as separate subprocesses with bounded parallelism; diagnostics and the
PDF build remain strictly downstream of successful completion of every run.

## 11. Provenance and stale-artifact validation

Every theory run records the code revision/dirty flag, unique slug and hierarchy,
full model definition, taxonomy, exact restrictions, maintained laws, domain and
path conditions, moving-reference flag, structural/observation frequencies,
inflation and expectation transformations/horizon/date, activity/competition
proxies, N0, zeta/mu treatment, sample, sampler, chains/seeds, and convergence
status. A SHA-256 signature covers the fields that define table comparability.

The restriction-report loader scans only the theory namespace, verifies the signature
and registry definition, and requires a single code revision, sample,
transformation, observation design, activity mapping, competition proxy, and
prior across a comparison table. A mismatch raises `StaleArtifactError`. Tests
also verify that changing one signed field is detected and that R1--R3 reject the
legacy direct YoY likelihood.

## 12. Verification status

- Historical model list and slugs remain unchanged.
- Existing unit, prior, transform, wrapper, FFBS, Particle-Gibbs, reporting, and
  robustness tests pass unchanged.
- New tests verify the cross-equation units, hierarchy, no-`b_x` guard, YoY guard,
  new posterior naming, weak Third law, and stale signature failure.
- Both reports were compiled independently with XeLaTeX; restriction-report
  pages were rendered and visually inspected without modifying the historical
  report.
- No legacy result was copied into the current theory tables. Production
  posterior estimates are intentionally pending; long MCMC runs were not
  fabricated or replaced by quick-run output.
