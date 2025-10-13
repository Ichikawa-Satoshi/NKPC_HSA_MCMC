import numpy as np
import matplotlib.pyplot as plt

def plot_trace(runs, burnin=1000, isvar=True, param_names=None, var_name=r'$\sigma^2$'):
    runs = np.asarray(runs)
    if runs.ndim != 2:
        raise ValueError("runs must be a 2D array (n_draws, n_params).")
    if burnin >= runs.shape[0]:
        raise ValueError("burnin must be smaller than the number of draws.")
    traces_all = runs[burnin:, :]
    n_remain, k = traces_all.shape

    if param_names is None:
        names = [f"beta_{i}" for i in range(k - (1 if isvar else 0))]
        if isvar:
            names.append(var_name)
        param_names = names
    else:
        if len(param_names) != k:
            raise ValueError("len(param_names) must match runs.shape[1].")
    
    fig, ax = plt.subplots(k, 1, figsize=(8, 1.5 * k), facecolor='w', squeeze=False)

    for idx in range(k):
        mc_trace = traces_all[:, idx]
        ax[idx, 0].plot(mc_trace, linewidth=0.4)
        ax[idx, 0].set_xlim(1, n_remain)
        ax[idx, 0].set_ylabel(param_names[idx])  #
        if idx == k - 1:
            ax[idx, 0].set_xlabel('samples')

    plt.tight_layout()
    plt.show()
    return