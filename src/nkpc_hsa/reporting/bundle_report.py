"""Standardized per-experiment result PDF from a bundle's saved draws.

Reads the ``.npz`` fits an experiment bundle wrote with
``nkpc_hsa.phillips.estimation._save_fit`` (under ``<results>/draws/``) and, when
present, a state file with ``qbar``/``qhat`` competition-state draws, and builds a
one-document-per-bundle report with the same six blocks the production
``each_result`` PDF carries:

1. posterior coefficient table,
2. prior-vs-posterior panels for every parameter,
3. time-varying ``kappa_t`` / ``theta_t`` paths (needs saved state),
4. competition-state decomposition ``qbar`` vs ``qhat`` (needs saved state),
5. a precision / spec comparison table,
6. R-hat and bulk-ESS convergence.

Blocks 3 and 4 are drawn only when the bundle saved a companion state file
(``<results>/draws/state.npz`` or ``<results>/state.npz`` with ``qbar``/``qhat``
arrays shaped ``chain x draw x T``); otherwise the report notes they were skipped.

Run it directly:

    python -m nkpc_hsa.reporting.bundle_report tests/<name>/results --compile
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from nkpc_hsa.config import load_yaml  # noqa: E402
from nkpc_hsa.reporting.tables import write_latex_fragment  # noqa: E402

_LABELS = {
    "a": r"$a$", "beta_b": r"$\beta_b$", "beta_f": r"$\beta_f$", "psi": r"$\psi$",
    "kappa_0": r"$\kappa_0$", "kappa_1": r"$\kappa_1$", "theta_0": r"$\theta_0$",
    "gamma": r"$\gamma$", "sigma": r"$\sigma$",
}


def _label(name: str) -> str:
    return _LABELS.get(name, name.replace("_", r"\_"))


@dataclass(frozen=True)
class FitData:
    label: str
    names: tuple[str, ...]
    coefficients: np.ndarray  # (chain, draw, param)
    sigma: np.ndarray  # (chain, draw)
    prior_sds: dict[str, float]
    q0: float
    model: str = ""
    estimator: str = ""
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    def draws(self, name: str) -> np.ndarray | None:
        if name == "sigma":
            return self.sigma.reshape(-1)
        if name in self.names:
            return self.coefficients[:, :, self.names.index(name)].reshape(-1)
        return None


def load_fits(results_dir: Path) -> list[FitData]:
    """Find every ``_save_fit`` npz anywhere under a bundle's ``results/``.

    Bundles save fits to different subdirectories (``draws/``, ``posterior/`` …),
    so we search recursively and keep any npz that carries ``coefficients`` +
    ``coefficient_names``. The per-fit label includes the parent directory when it
    is not ``draws``/``posterior``, so runs in different subfolders stay distinct.
    """
    fits: list[FitData] = []
    for path in sorted(Path(results_dir).rglob("*.npz")):
        if path.name == "state.npz":
            continue
        data = np.load(path, allow_pickle=True)
        if "coefficients" not in data or "coefficient_names" not in data:
            continue
        label = path.stem
        if path.parent.name not in {"draws", "posterior"}:
            label = f"{path.parent.name}/{label}"
        names = tuple(str(n) for n in data["coefficient_names"])
        priors = {}
        if "prior_sd_names" in data and "prior_sd_values" in data:
            priors = {str(n): float(v) for n, v in zip(data["prior_sd_names"], data["prior_sd_values"])}
        reserved = {
            "coefficients", "sigma", "coefficient_names", "prior_sd_names", "prior_sd_values",
            "q0", "x_scale", "cell", "inflation", "activity", "model", "estimator",
        }
        aux = {k: data[k] for k in data.files if k not in reserved and data[k].ndim == 2}
        fits.append(
            FitData(
                label=label,
                names=names,
                coefficients=np.asarray(data["coefficients"], float),
                sigma=np.asarray(data["sigma"], float),
                prior_sds=priors,
                q0=float(data["q0"]) if "q0" in data else 0.0,
                model=str(data["model"]) if "model" in data else "",
                estimator=str(data["estimator"]) if "estimator" in data else "",
                aux=aux,
            )
        )
    return fits


# ---------------------------------------------------------------- model equation


_TERMS = {
    "beta_b": (r"\beta_b\,\pi_{t-1}", "+"),
    "beta_f": (r"\beta_f\,\mathbb{E}_t\pi_{t+1}", "+"),
    "psi": (r"\psi\,(\bar N_t-N_0)", "+"),
    "kappa_0": (r"\kappa_0\,x_t", "+"),
    "kappa_1": (r"\kappa_1\,(\bar N_t-N_0)\,x_t", "+"),
    "theta_0": (r"\theta_0\,\hat N_t", "-"),
    "gamma": (r"\gamma\,(\bar N_t-N_0)\,\hat N_t", "-"),
    "theta_hsa": (r"\theta_{\mathrm{hsa}}\,\big[b_x\zeta(\bar N_t-N_0)x_t-\hat N_t\big]", "+"),
    "lambda_qx": (r"\lambda_{qx}\,\hat N_t x_t", "+"),
    "lambda_u": (r"\lambda_u\,u_t", "+"),
    "lambda_eta": (r"\lambda_\eta\,\eta_t", "+"),
}


def model_equation(names: tuple[str, ...]) -> str:
    """Reconstruct the estimated linear specification as a LaTeX equation.

    The regressor for each coefficient follows the shared design built by
    ``nkpc_hsa.phillips.inflation._quarterly_design`` (``theta_0`` / ``gamma``
    enter with a minus sign). Terms not in the map fall back to a generic label.
    """
    rhs = "a" if "a" in names else ""
    for name in names:
        if name == "a":
            continue
        term, sign = _TERMS.get(name, (rf"\beta_{{\mathrm{{{name.replace('_', chr(92) + '_')}}}}}\,z_t", "+"))
        rhs = f"{term}" if rhs == "" else f"{rhs} {sign} {term}"
    return r"\pi_t = " + rhs + r" + \varepsilon_t"


def _restriction_note(estimator: str) -> str:
    if "adding_up" in estimator:
        return r" \quad\text{(restriction: } \beta_b+\beta_f=1,\ \beta_f\in[0,1]\text{)}"
    if "convexity" in estimator:
        return r" \quad\text{(restriction: } \beta_b,\beta_f\in[0,1]\text{)}"
    if "persistent_ar1" in estimator:
        return r" \quad\text{(error: } \varepsilon_t=\rho\,\varepsilon_{t-1}+\nu_t\text{)}"
    if "low_frequency" in estimator:
        return r" \quad\text{(error: persistent low-frequency component)}"
    return ""


def load_state(results_dir: Path) -> dict | None:
    """Competition-state draws for blocks 3-4: a ``state.npz`` with qbar/qhat."""
    for candidate in sorted(Path(results_dir).rglob("state.npz")):
        data = np.load(candidate, allow_pickle=True)
        if "qbar" in data and "qhat" in data:
            return {
                "qbar": np.asarray(data["qbar"], float),
                "qhat": np.asarray(data["qhat"], float),
                "q0": float(data["q0"]) if "q0" in data else 0.0,
                "periods": [str(p) for p in data["periods"]] if "periods" in data else None,
            }
    return None


def load_spec(results_dir: Path) -> dict | None:
    """Optional per-experiment ``spec.yaml`` describing equation / data / priors.

    Looked for in the bundle directory (``tests/<name>/spec.yaml``, i.e.
    ``results_dir.parent``) and in ``results_dir`` itself. Any of these keys are
    honoured (all optional): ``description`` (prose intro), ``equation`` (LaTeX,
    no ``$``; overrides the auto-derived one), ``data`` (list of
    ``{series, source, transform}``), ``priors`` (``{param: "$...$"}``).
    """
    for base in (Path(results_dir).parent, Path(results_dir)):
        path = base / "spec.yaml"
        if path.exists():
            try:
                return load_yaml(path) or {}
            except Exception:
                return None
    return None


def data_table(spec: dict) -> pd.DataFrame | None:
    rows = spec.get("data") or []
    if not rows:
        return None
    return pd.DataFrame(
        [
            {
                "Series": str(r.get("series", "")),
                "Source": str(r.get("source", "")),
                "Transform": str(r.get("transform", "")),
            }
            for r in rows
        ]
    )


def priors_table(spec: dict | None, fits: list["FitData"]) -> pd.DataFrame:
    """One row per parameter: the prior from spec.yaml, else the saved SD."""
    order = ["a", "beta_b", "beta_f", "psi", "kappa_0", "kappa_1", "theta_0", "gamma", "sigma"]
    present = [n for n in order if any(f.draws(n) is not None for f in fits)]
    spec_priors = (spec or {}).get("priors", {}) or {}
    saved = fits[0].prior_sds if fits else {}
    rows = []
    for name in present:
        described = spec_priors.get(name)
        if described is None:
            if name == "sigma":
                described = r"$\sigma^2\sim\mathrm{IG}(3,\,8)$"
            elif name in saved:
                described = rf"$\mathcal{{N}}(0,\;{saved[name]:.3g}^2)$"
            else:
                described = "---"
        rows.append({"Parameter": _label(name), "Prior": str(described)})
    return pd.DataFrame(rows)


def _esc(text: str) -> str:
    return text.replace("_", r"\_")


def _estimate(values: np.ndarray) -> str:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return f"\\makecell[r]{{{np.mean(values):.4f}\\\\\\scriptsize [{lo:.4f}, {hi:.4f}]}}"


# ---------------------------------------------------------------- tables (1, 5, 6)


def coefficient_table(fits: list[FitData]) -> pd.DataFrame:
    """Wide format: specs down the rows, parameters across the columns.

    Each cell is a compact two-line ``mean`` over ``(2.5%, 97.5%)`` so the table
    stays readable; it is typeset inside a ``\\resizebox`` to fit the page width.
    """
    order = ["a", "beta_b", "beta_f", "psi", "kappa_0", "kappa_1", "theta_0", "gamma", "sigma"]
    present = [n for n in order if any(f.draws(n) is not None for f in fits)]
    rows = []
    for fit in fits:
        row = {"Spec": _esc(fit.label)}
        for name in present:
            draws = fit.draws(name)
            if draws is None:
                row[_label(name)] = ""
            else:
                lo, hi = np.quantile(draws, [0.025, 0.975])
                row[_label(name)] = (
                    f"\\makecell{{{float(np.mean(draws)):.3f}\\\\\\scriptsize({lo:.3f}, {hi:.3f})}}"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def precision_table(fits: list[FitData]) -> pd.DataFrame:
    rows = []
    for fit in fits:
        row = {"Spec": _esc(fit.label), "$\\sigma$ (resid.)": _estimate(fit.sigma.reshape(-1))}
        kappa = fit.draws("kappa_1")
        if kappa is not None:
            row["$\\kappa_1$"] = _estimate(kappa)
        for key, arr in fit.aux.items():
            row[_esc(key.replace("_", " "))] = f"{float(np.mean(arr)):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_table(fits: list[FitData]) -> pd.DataFrame:
    """Worst R-hat and smallest bulk ESS per spec (arviz, per scalar parameter)."""
    import arviz as az

    rows = []
    for fit in fits:
        arrays = [fit.coefficients[:, :, i] for i in range(len(fit.names))] + [fit.sigma]
        rhats, esss = [], []
        for arr in arrays:
            try:
                rh, es = float(az.rhat(arr)), float(az.ess(arr))
            except (ValueError, TypeError):
                continue
            if np.isfinite(rh):
                rhats.append(rh)
            if np.isfinite(es):
                esss.append(es)
        rows.append(
            {
                "Spec": _esc(fit.label),
                "Max $\\hat{R}$": f"{max(rhats):.3f}" if rhats else "",
                "Min bulk ESS": f"{min(esss):.0f}" if esss else "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures (2, 3, 4)


def _prior_curve(name: str, sd: float, grid: np.ndarray) -> np.ndarray:
    if name == "sigma":
        # sigma^2 ~ InverseGamma(3, 8) (the estimation default); density on sigma.
        a, b = 3.0, 8.0
        s2 = np.clip(grid, 1e-9, None) ** 2
        log_ig = a * np.log(b) - math.lgamma(a) - (a + 1) * np.log(s2) - b / s2
        return 2.0 * np.clip(grid, 1e-9, None) * np.exp(log_ig)
    return np.exp(-0.5 * (grid / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def _smooth_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    """Gaussian-KDE posterior density; None if the draws are (near-)degenerate."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 5 or float(np.std(values)) < 1e-9:
        return None
    try:
        return gaussian_kde(values)(grid)
    except (np.linalg.LinAlgError, ValueError):
        return None


def plot_prior_posterior(fit: FitData, path: Path) -> Path | None:
    names = [n for n in (*fit.names, "sigma")]
    names = [n for n in names if fit.draws(n) is not None]
    if not names:
        return None
    ncol = min(len(names), 5)  # wide, short grid suits the landscape page
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, name in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        values = fit.draws(name)
        lo, hi = np.quantile(values, [0.001, 0.999])
        grid = np.linspace(lo, hi, 256)
        density = _smooth_density(values, grid)
        if density is not None:
            ax.fill_between(grid, density, color="#4477aa", alpha=0.45)
            ax.plot(grid, density, color="#33557a", lw=1.2, label="posterior")
        else:  # ~degenerate draw: mark the point mass instead of a KDE
            ax.axvline(float(np.mean(values)), color="#33557a", lw=1.2, label="posterior")
        sd = fit.prior_sds.get(name)
        if name == "sigma" or sd is not None:
            curve = _prior_curve(name, sd or 1.0, grid)
            if np.any(np.isfinite(curve)):
                ax.plot(grid, curve, color="#cc3311", lw=1.4, ls="--", label="prior")
        ax.set_title(_label(name))
        ax.set_yticks([])
    axes[0][0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Prior vs posterior — {fit.label}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _time_axis(state: dict, length: int):
    if state.get("periods") and len(state["periods"]) == length:
        try:
            return pd.PeriodIndex(state["periods"], freq="Q").to_timestamp()
        except (ValueError, TypeError):
            pass
    return np.arange(length)


def plot_kappa_paths(fits: list[FitData], state: dict, path: Path) -> Path | None:
    """Small multiples: one panel per specification, a single kappa_t line each.

    (Time-series figures never overlay 3+ lines; each panel shows one median line
    with its 90% band.)
    """
    qbar = state["qbar"]
    if qbar.ndim != 3:
        return None
    centered = qbar - state["q0"]
    x = _time_axis(state, qbar.shape[-1])
    eligible = [
        f for f in fits
        if f.coefficients.shape[:2] == qbar.shape[:2] and "kappa_1" in f.names
    ]
    if not eligible:
        return None
    ncol = min(len(eligible), 3)
    nrow = int(np.ceil(len(eligible) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.5 * nrow), squeeze=False, sharex=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, fit in enumerate(eligible):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        k0 = fit.coefficients[:, :, fit.names.index("kappa_0")][:, :, None]
        k1 = fit.coefficients[:, :, fit.names.index("kappa_1")][:, :, None]
        kappa_t = (k0 + k1 * centered).reshape(-1, qbar.shape[-1])
        med = np.median(kappa_t, axis=0)
        lo, hi = np.quantile(kappa_t, [0.05, 0.95], axis=0)
        ax.plot(x, med, color="#0072B2", lw=1.5)
        ax.fill_between(x, lo, hi, color="#0072B2", alpha=0.18)
        ax.axhline(0.0, color="black", lw=0.7, ls="--")
        ax.set_title(_esc(fit.label), fontsize=8)
    fig.suptitle(r"Time-varying slope $\kappa_t=\kappa_0+\kappa_1(\bar N_t-N_0)$", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_decomposition(state: dict, path: Path) -> Path | None:
    """Two lines only: the slow trend qbar and the fast component qhat."""
    qbar, qhat = state["qbar"], state["qhat"]
    if qbar.ndim != 3:
        return None
    x = _time_axis(state, qbar.shape[-1])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for arr, name, color in ((qbar, r"$\bar N_t$ (slow trend)", "#0072B2"), (qhat, r"$\hat N_t$ (fast)", "#D55E00")):
        flat = arr.reshape(-1, arr.shape[-1])
        med = np.median(flat, axis=0)
        lo, hi = np.quantile(flat, [0.05, 0.95], axis=0)
        ax.plot(x, med, color=color, lw=1.5, label=name)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    ax.axhline(0.0, color="black", lw=0.7, ls="--")
    ax.set_ylabel("competition state (ten-log-point units)")
    ax.set_title("Competition-state decomposition")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- assembly


_TEMPLATE = r"""\documentclass[11pt,landscape]{article}
\usepackage[margin=1.2cm]{geometry}
\usepackage{booktabs,graphicx,makecell,amsmath,amssymb,adjustbox}
\graphicspath{{figures/}}
\setlength{\parindent}{0pt}
% Figures are height- and width-bounded so nothing runs off the page. A WIDE
% table is shrunk to the line width (\fitwide); a NARROW table is left at its
% natural size (\fitnarrow) so it is not blown up past the page height. The row
% count is capped (--max-fits) so neither can grow taller than a page.
\newcommand{\fitfig}[1]{\begin{center}%
\includegraphics[width=\linewidth,height=0.78\textheight,keepaspectratio]{#1}%
\end{center}}
% Shrink a table to the line width ONLY when it is too wide; never enlarge a
% narrow table (which would blow up its height). So no table can overflow.
\newcommand{\fittable}[1]{\begin{center}%
\adjustbox{max width=\linewidth,max totalheight=0.86\textheight}{\input{#1}}%
\end{center}}
\begin{document}
\begin{center}\Large\textbf{%(title)s}\end{center}
\medskip
%(body)s
\end{document}
"""


def build_report(
    results_dir: Path, out_dir: Path | None = None, *, compile_pdf: bool = False, max_fits: int = 16
) -> Path:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"{results_dir} does not exist; run the experiment first.")
    all_fits = load_fits(results_dir)
    if not all_fits:
        raise FileNotFoundError(
            f"No fit .npz files (with 'coefficients') found under {results_dir}. "
            "Run the experiment's run.py first."
        )
    truncated = len(all_fits) > max_fits
    fits = all_fits[:max_fits]  # bound the report for large grid bundles
    state = load_state(results_dir)
    spec = load_spec(results_dir)
    out = Path(out_dir) if out_dir is not None else results_dir / "report"
    tables, figures = out / "tables", out / "figures"
    for directory in (tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    body: list[str] = []
    if spec and spec.get("description"):
        body.append(str(spec["description"]).strip() + r"\par\medskip")
    if truncated:
        body.append(
            rf"\textit{{Showing the first {len(fits)} of {len(all_fits)} saved "
            rf"specifications (raise \texttt{{--max-fits}} to include more).}}\par\medskip"
        )

    # Block 0: the estimated equation — from spec.yaml if given, else auto-derived.
    body.append(r"\section*{0. Estimated specification}")
    _fit_eq = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}"
    if spec and spec.get("equation"):
        body.append(r"\begin{center}" + _fit_eq + r"{$\displaystyle " + str(spec["equation"]).strip() + r"$}\end{center}")
    else:
        for names_sig, note in dict.fromkeys((fit.names, _restriction_note(fit.estimator)) for fit in fits):
            body.append(r"\begin{center}" + _fit_eq + r"{$\displaystyle " + model_equation(names_sig) + note + r"$}\end{center}")

    # Data block (only when spec.yaml lists data series).
    _data = data_table(spec) if spec else None
    if _data is not None:
        write_latex_fragment(_data, tables / "data.tex", escape=False)
        body.append(r"\section*{0b. Data}")
        body.append(r"\fittable{tables/data.tex}")

    # Priors block (spec descriptions, else the saved prior SDs).
    write_latex_fragment(priors_table(spec, fits), tables / "priors.tex", escape=False)
    body.append(r"\section*{0c. Priors}")
    body.append(r"\fittable{tables/priors.tex}")

    write_latex_fragment(coefficient_table(fits), tables / "coefficients.tex", escape=False)
    body.append(r"\section*{1. Posterior coefficients --- mean over (2.5\%, 97.5\%)}")
    body.append(r"\fittable{tables/coefficients.tex}")

    body.append(r"\section*{2. Prior vs posterior (KDE)}")
    for fit in fits:
        name = f"prior_posterior_{fit.label.replace('/', '__')}.png"
        if plot_prior_posterior(fit, figures / name) is not None:
            body.append(f"\\fitfig{{figures/{name}}}")

    body.append(r"\section*{3--4. Time-varying slope and competition-state decomposition}")
    if state is not None:
        if plot_kappa_paths(fits, state, figures / "kappa_paths.png") is not None:
            body.append(r"\fitfig{figures/kappa_paths.png}")
        if plot_decomposition(state, figures / "decomposition.png") is not None:
            body.append(r"\fitfig{figures/decomposition.png}")
    else:
        body.append(
            r"No competition-state file (\texttt{state.npz} with \texttt{qbar}/\texttt{qhat}) was "
            r"saved with this bundle, so the time-varying slope and decomposition are omitted."
        )

    write_latex_fragment(precision_table(fits), tables / "precision.tex", escape=False)
    body.append(r"\section*{5. Precision / specification comparison}")
    body.append(r"\fittable{tables/precision.tex}")
    write_latex_fragment(convergence_table(fits), tables / "convergence.tex", escape=False)
    body.append(r"\section*{6. Convergence: max $\hat R$ and min bulk ESS}")
    body.append(r"\fittable{tables/convergence.tex}")

    # Bundle case: .../<name>/results -> use <name>; shared case: results/<name> -> use <name>.
    bundle_name = results_dir.parent.name if results_dir.name == "results" else results_dir.name
    title = f"Experiment result report --- {bundle_name.replace('_', chr(92) + '_')}"
    tex_path = out / f"{bundle_name}_report.tex"
    document = _TEMPLATE.replace("%(title)s", title).replace("%(body)s", "\n".join(body))
    tex_path.write_text(document, encoding="utf-8")

    if compile_pdf:
        result = None
        for _ in (1, 2):
            result = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=tex_path.parent, capture_output=True, text=True,
            )
        pdf = tex_path.with_suffix(".pdf")
        if result is None or result.returncode != 0 or not pdf.exists():
            tail = "\n".join((result.stdout if result else "").splitlines()[-40:])
            raise SystemExit(f"xelatex failed:\n{tail}")
        return pdf
    return tex_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 6-block PDF from an experiment bundle's saved draws.")
    parser.add_argument("results_dir", type=Path, help="A bundle results/ directory (holds draws/).")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--compile", action="store_true", help="Run xelatex to produce the PDF.")
    parser.add_argument("--max-fits", type=int, default=16, help="Cap specifications shown (default 16).")
    args = parser.parse_args()
    target = build_report(args.results_dir, args.out_dir, compile_pdf=args.compile, max_fits=args.max_fits)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
