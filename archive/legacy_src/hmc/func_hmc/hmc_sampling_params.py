def hmc_sampling_params(
    warmup=5000,
    samples=20000,
    chains=2,
    target_accept=0.95,
    chain_method="parallel",
    progress_bar=True
):
    """
    Select MCMC sampling parameters.

    Parameters:
    - warmup: Number of warmup samples.
    - samples: Number of sampling samples.
    - chains: Number of chains.
    - target_accept: Target acceptance probability for NUTS.
    - chain_method: Chain method ("parallel", "sequential", etc.).
    - progress_bar: Whether to show progress bar.

    Returns:
    - dict: Dictionary of sampling parameters.
    """
    return {
        'warmup': warmup,
        'samples': samples,
        'chains': chains,
        'target_accept': target_accept,
        'chain_method': chain_method,
        'progress_bar': progress_bar,
    }
