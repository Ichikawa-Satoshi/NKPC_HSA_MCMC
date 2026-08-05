from __future__ import annotations

from pathlib import Path


def plot_competition_path_comparison(
    quarterly_index,
    Nbar_draws,
    Nhat_draws,
    N_obs_used,
    N_annual_q4,
    N_interpolated_comparison,
    model_name: str,
    activity_name: str,
    frequency: str,
    output_dir: str | Path,
    credible_interval: float = 0.90,
) -> Path:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    q_index = quarterly_index
    if isinstance(q_index, pd.PeriodIndex):
        periods = q_index.asfreq("Q")
    else:
        periods = pd.Index(q_index)
        if np.issubdtype(periods.dtype, np.datetime64):
            periods = pd.DatetimeIndex(periods).to_period("Q")
        else:
            periods = pd.period_range("2000Q1", periods=np.asarray(N_obs_used).size, freq="Q")
    dates = periods.to_timestamp(how="end")

    nbar = np.asarray(Nbar_draws, dtype=float)
    nhat = np.asarray(Nhat_draws, dtype=float)
    if nbar.shape != nhat.shape or nbar.ndim < 2:
        raise ValueError("Nbar_draws and Nhat_draws must have the same shape and include a time axis.")
    total = (nbar + nhat).reshape(-1, nbar.shape[-1])
    center = np.nanmedian(total, axis=0)
    alpha = (1.0 - credible_interval) / 2.0
    lo = np.nanquantile(total, alpha, axis=0)
    hi = np.nanquantile(total, 1.0 - alpha, axis=0)
    nbar_center = np.nanmedian(nbar.reshape(-1, nbar.shape[-1]), axis=0)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe = "_".join(
        part.replace("/", "-")
        for part in [model_name, activity_name, frequency]
        if part
    )
    out_path = target_dir / f"competition_path_comparison_{safe}.png"

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(dates, center, label="latent N total posterior median", color="C0", lw=1.8)
    ax.fill_between(dates, lo, hi, color="C0", alpha=0.16, label=f"{int(credible_interval * 100)}% credible interval")

    if N_interpolated_comparison is not None:
        comp = np.asarray(N_interpolated_comparison, dtype=float).reshape(-1)
        if comp.size == center.size:
            label = "Q4-anchored PCHIP comparison" if frequency == "annual_q4" else "PCHIP quarterly N_obs"
            ax.plot(dates, comp, label=label, color="0.25", ls="--", lw=1.3)

    annual = None if N_annual_q4 is None else np.asarray(N_annual_q4, dtype=float).reshape(-1)
    if annual is None:
        annual = np.asarray(N_obs_used, dtype=float).reshape(-1)
    if annual.size == center.size:
        mask = np.isfinite(annual)
        if mask.any():
            ax.scatter(dates[mask], annual[mask], label="annual Q4 observations", color="C3", s=24, zorder=4)

    ax.plot(dates, nbar_center, label="Nbar posterior median", color="C2", lw=1.1, alpha=0.85)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.45)
    ax.set_title(f"Competition path comparison: {model_name} / {activity_name} / {frequency}")
    ax.set_xlabel("quarter")
    ax.set_ylabel("N units")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


