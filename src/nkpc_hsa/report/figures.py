from __future__ import annotations

from pathlib import Path


def save_placeholder_figure(out_path: str | Path, message: str) -> None:
    import matplotlib.pyplot as plt

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_posterior_density(idata, var_name: str, out_path: str | Path) -> None:
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(az, "plot_posterior"):
        az.plot_posterior(idata, var_names=[var_name], hdi_prob=0.95)
    else:
        values = np.asarray(idata.posterior[var_name], dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(values, bins=30, density=True, alpha=0.65)
        ax.axvline(np.mean(values), color="black", lw=1.0)
        ax.set_title(f"Posterior density: {var_name}")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(target, dpi=200, bbox_inches="tight")
    plt.close("all")


def save_posterior_predictive_placeholder(out_path: str | Path) -> None:
    save_placeholder_figure(out_path, "Posterior predictive checks are generated after model runs.")


def _posterior_values(idata, var_name: str):
    import numpy as np

    posterior = getattr(idata, "posterior", None)
    if posterior is None or var_name not in posterior:
        return None
    values = np.asarray(posterior[var_name], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return values if values.size else None


def save_prior_posterior_overlay(
    idata,
    var_name: str,
    prior: tuple[float, float],
    out_path: str | Path,
    *,
    scale: float = 1.0,
    display_name: str | None = None,
    xlabel: str | None = None,
    zoom_to_posterior: bool = False,
) -> bool:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde, norm

    values = _posterior_values(idata, var_name)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if values is None or values.size < 3 or float(np.std(values, ddof=1)) <= 0.0:
        return False
    mu, sd = prior
    values = values * scale
    mu = mu * scale
    sd = abs(sd * scale)
    if zoom_to_posterior:
        center = float(np.median(values))
        spread = max(float(np.std(values, ddof=1)), 1e-8)
        lo = min(float(np.quantile(values, 0.005)), center - 6.0 * spread)
        hi = max(float(np.quantile(values, 0.995)), center + 6.0 * spread)
    else:
        lo = min(float(np.quantile(values, 0.005)), mu - 4.0 * sd)
        hi = max(float(np.quantile(values, 0.995)), mu + 4.0 * sd)
    xs = np.linspace(lo, hi, 400)
    kde = gaussian_kde(values)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sd), label="prior", color="0.35", linestyle="--")
    ax.plot(xs, kde(xs), label="posterior", color="C0")
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(f"Prior vs posterior: {display_name or var_name}")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_kappa_model_comparison(kappa_table, out_path: str | Path) -> bool:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if kappa_table is None or kappa_table.empty:
        return False
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in kappa_table.iterrows():
        mean = row.get("kappa_mean")
        label = "kappa"
        if pd.isna(mean):
            mean = row.get("kappa_t_overall_mean", row.get("kappa_0_mean"))
            label = "kappa_t avg" if "kappa_t_overall_mean" in row and pd.notna(row.get("kappa_t_overall_mean")) else "kappa_0"
        if pd.notna(mean):
            rows.append((str(row.get("model", row.get("run", ""))), str(row.get("run", "")), label, float(mean)))
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.45 * len(rows))))
    labels = [f"{model}\n{label}" for model, _, label, _ in rows]
    means = [mean for *_, mean in rows]
    ypos = np.arange(len(rows))
    ax.barh(ypos, means, color="C0", alpha=0.75)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("posterior mean")
    ax.set_title("Kappa comparison across models")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_coefficient_interval_plot(
    table,
    out_path: str | Path,
    *,
    title: str,
    label_columns: tuple[str, ...] = ("model", "parameter"),
    max_rows: int = 48,
) -> bool:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if table is None or table.empty:
        return False
    needed = {"posterior_mean", "ci_2.5", "ci_97.5"}
    if not needed <= set(table.columns):
        return False
    df = table.copy()
    df = df.dropna(subset=["posterior_mean", "ci_2.5", "ci_97.5"])
    if df.empty:
        return False
    if len(df) > max_rows:
        df = df.head(max_rows)
    labels = []
    for _, row in df.iterrows():
        parts = [str(row[col]) for col in label_columns if col in row and pd.notna(row[col]) and str(row[col])]
        labels.append(" | ".join(parts))
    mean = df["posterior_mean"].to_numpy(dtype=float)
    lo = df["ci_2.5"].to_numpy(dtype=float)
    hi = df["ci_97.5"].to_numpy(dtype=float)
    y = np.arange(len(df))
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.32 * len(df))))
    ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="o", ms=4, capsize=2, color="C0", ecolor="0.45")
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("posterior mean and 95% interval")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_time_varying_path(idata_by_run: dict[str, object], var_name: str, out_path: str | Path) -> bool:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    plotted = False
    sample_start: str | None = None
    for run, idata in idata_by_run.items():
        posterior = getattr(idata, "posterior", None)
        if posterior is None or var_name not in posterior:
            continue
        arr = np.asarray(posterior[var_name], dtype=float)
        if arr.ndim < 3:
            continue
        path = arr.reshape(-1, arr.shape[-1])
        mean = np.nanmean(path, axis=0)
        lo = np.nanquantile(path, 0.025, axis=0)
        hi = np.nanquantile(path, 0.975, axis=0)
        if sample_start is None:
            sample_start = str(getattr(idata, "attrs", {}).get("sample_start", "") or "")
        if sample_start:
            x = pd.date_range(start=sample_start, periods=mean.size, freq="QE")
        else:
            x = np.arange(mean.size)
        ax.plot(x, mean, label=str(getattr(idata, "attrs", {}).get("model", run)))
        ax.fill_between(x, lo, hi, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(f"Time-varying coefficient path: {var_name}")
    if sample_start:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate(rotation=0, ha="center")
        ax.set_xlabel("year")
    else:
        ax.set_xlabel("time index")
    ax.set_ylabel("posterior mean and 95% interval")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_prior_posterior_per_model(
    idata_by_run: dict[str, object],
    parameters: list[str],
    prior_getter,
    out_dir: str | Path,
    overlay_options: dict[str, dict] | None = None,
) -> dict[str, Path]:
    """One figure per model, grid of subplots (one per parameter with draws).

    Returns {model_name: figure_path}. ``prior_getter(idata, var)`` returns
    ``(mu, sd)`` or ``None``.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde, norm

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opts = overlay_options or {}
    result: dict[str, Path] = {}

    by_model: dict[str, object] = {}
    for idata in idata_by_run.values():
        model = str(getattr(idata, "attrs", {}).get("model", "unknown"))
        if model not in by_model:
            by_model[model] = idata

    model_order = ["ces", "hsa_steady", "hsa_dynamic", "hsa_full"]
    ordered = [(m, by_model[m]) for m in model_order if m in by_model]
    ordered += [(m, v) for m, v in by_model.items() if m not in model_order]

    for model, idata in ordered:
        posterior = getattr(idata, "posterior", None)
        if posterior is None:
            continue
        var_data = []
        for var in parameters:
            if var not in posterior:
                continue
            values = np.asarray(posterior[var], dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size < 3 or float(np.std(values, ddof=1)) <= 0.0:
                continue
            prior_tuple = prior_getter(idata, var)
            if prior_tuple is None:
                continue
            var_data.append((var, values, prior_tuple))
        if not var_data:
            continue
        ncols = min(3, len(var_data))
        nrows = (len(var_data) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
        for i, (var, values, (mu, sd)) in enumerate(var_data):
            ax = axes[i // ncols][i % ncols]
            opt = opts.get(var, {})
            zoom = opt.get("zoom_to_posterior", False)
            if zoom:
                center = float(np.median(values))
                spread = max(float(np.std(values, ddof=1)), 1e-8)
                lo = min(float(np.quantile(values, 0.005)), center - 6.0 * spread)
                hi = max(float(np.quantile(values, 0.995)), center + 6.0 * spread)
            else:
                lo = min(float(np.quantile(values, 0.005)), mu - 4.0 * sd)
                hi = max(float(np.quantile(values, 0.995)), mu + 4.0 * sd)
            xs = np.linspace(lo, hi, 400)
            kde = gaussian_kde(values)
            ax.plot(xs, norm.pdf(xs, loc=mu, scale=sd), label="prior", color="0.35", linestyle="--")
            ax.plot(xs, kde(xs), label="posterior", color="C0")
            ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
            ax.set_title(var, fontsize=10)
            if xlabel := opt.get("xlabel"):
                ax.set_xlabel(xlabel, fontsize=8)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.25)
        for i in range(len(var_data), nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)
        fig.suptitle(f"Prior vs posterior — {model}", fontsize=12, fontweight="bold", y=1.01)
        fig.tight_layout()
        out_path = out / f"prior_posterior_{model}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        result[model] = out_path

    return result
