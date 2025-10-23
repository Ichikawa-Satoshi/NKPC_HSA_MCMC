function results = func_nkpc_ces(pi_data, pi_prev_data, Epi_data, x_data, n_burn, n_keep, priors, opts)
% Estimates NKPC CES parameters (alpha, kappa, sigma_v^2) via Gibbs sampling
% Priors and options are supplied externally.
%
% priors fields (all optional, defaults in parentheses):
%   mu_alpha(0.5), sigma_alpha(0.2)
%   mu_kappa(0.0), sigma_kappa(0.5)
%   a_sig(2.0), b_sig(2.0)   % Inv-Gamma(shape=a_sig, scale=b_sig)
%
% opts fields (all optional):
%   alpha0(0.5), kappa0(0.1), sigma_v20(0.1)
%   seed([] -> no rng set), constrain_alpha(false), verbose(true), store_every(1)

    %% Data
    pi_t     = pi_data(:);
    pi_tm1   = pi_prev_data(:);
    E_pi_tp1 = Epi_data(:);
    x_t      = x_data(:);

    T_obs = numel(pi_t);
    if any([numel(pi_tm1), numel(E_pi_tp1), numel(x_t)] ~= T_obs)
        error('Input vectors must have the same length.');
    end

    %% Priors (with safe defaults)
    if nargin < 7 || isempty(priors), priors = struct(); end
    mu_alpha    = getfield_with_default(priors, 'mu_alpha',    0.5);
    sigma_alpha = getfield_with_default(priors, 'sigma_alpha', 0.2);   % std (not var)
    mu_kappa    = getfield_with_default(priors, 'mu_kappa',    0.0);
    sigma_kappa = getfield_with_default(priors, 'sigma_kappa', 0.5);   % std (not var)
    a_sig       = getfield_with_default(priors, 'a_sig',       2.0);
    b_sig       = getfield_with_default(priors, 'b_sig',       2.0);

    if sigma_alpha <= 0 || sigma_kappa <= 0 || a_sig <= 0 || b_sig <= 0
        error('Prior hyperparameters must be positive where applicable.');
    end

    %% Options
    if nargin < 8 || isempty(opts), opts = struct(); end
    alpha  = getfield_with_default(opts, 'alpha0',    0.5);
    kappa  = getfield_with_default(opts, 'kappa0',    0.1);
    sigma_v2 = getfield_with_default(opts, 'sigma_v20', 0.1);
    seed   = getfield_with_default(opts, 'seed',      []);
    constrain_alpha = getfield_with_default(opts, 'constrain_alpha', false);
    verbose = getfield_with_default(opts, 'verbose', true);
    store_every = max(1, getfield_with_default(opts, 'store_every', 1));

    if ~isempty(seed), rng(seed); end

    if verbose
        fprintf('Initial: alpha=%.3f, kappa=%.3f, sigma_v2=%.3f\n', alpha, kappa, sigma_v2);
        fprintf('Burn-in: %d, Keep: %d (store every %d)\n', n_burn, n_keep, store_every);
    end

    %% Storage
    n_store = ceil(n_keep / store_every);
    alpha_draws  = zeros(n_store,1);
    kappa_draws  = zeros(n_store,1);
    sigma_draws  = zeros(n_store,1);

    store_idx = 0;

    %% Gibbs
    total_iter = n_burn + n_keep;
    for iter = 1:total_iter

        % ---- Sample alpha | kappa, sigma_v2, data
        % π_t - Eπ_{t+1} - κ x_t = α (π_{t-1} - Eπ_{t+1}) + v_t
        y_alpha = pi_t - E_pi_tp1 - kappa .* x_t;
        X_alpha = pi_tm1 - E_pi_tp1;

        prior_prec_a = 1/(sigma_alpha^2);
        data_prec_a  = (X_alpha' * X_alpha) / sigma_v2;
        post_prec_a  = prior_prec_a + data_prec_a;
        post_var_a   = 1 / post_prec_a;
        post_mean_a  = post_var_a * (prior_prec_a * mu_alpha + (X_alpha' * y_alpha) / sigma_v2);

        alpha_draw = post_mean_a + sqrt(post_var_a) * randn;

        if constrain_alpha
            % simple rejection to keep alpha in (0,1)
            max_trials = 10000;
            trials = 0;
            while (alpha_draw <= 0 || alpha_draw >= 1) && trials < max_trials
                alpha_draw = post_mean_a + sqrt(post_var_a) * randn;
                trials = trials + 1;
            end
            if trials == max_trials && verbose
                warning('alpha draw hit bounds frequently; consider a truncated normal sampler.');
            end
        end
        alpha = alpha_draw;

        % ---- Sample kappa | alpha, sigma_v2, data
        % π_t - α π_{t-1} - (1-α) Eπ_{t+1} = κ x_t + v_t
        y_kappa = pi_t - alpha .* pi_tm1 - (1 - alpha) .* E_pi_tp1;
        X_kappa = x_t;

        prior_prec_k = 1/(sigma_kappa^2);
        data_prec_k  = (X_kappa' * X_kappa) / sigma_v2;
        post_prec_k  = prior_prec_k + data_prec_k;
        post_var_k   = 1 / post_prec_k;
        post_mean_k  = post_var_k * (prior_prec_k * mu_kappa + (X_kappa' * y_kappa) / sigma_v2);

        kappa = post_mean_k + sqrt(post_var_k) * randn;

        % ---- Sample sigma_v^2 | alpha, kappa, data
        resid = pi_t - alpha .* pi_tm1 - (1 - alpha) .* E_pi_tp1 - kappa .* x_t;
        a_post = a_sig + T_obs/2;
        b_post = b_sig + 0.5 * sum(resid.^2);
        sigma_v2 = 1 / gamrnd(a_post, 1/b_post);

        % ---- Store after burn-in (with thinning)
        if iter > n_burn
            if mod(iter - n_burn, store_every) == 0
                store_idx = store_idx + 1;
                alpha_draws(store_idx) = alpha;
                kappa_draws(store_idx) = kappa;
                sigma_draws(store_idx) = sigma_v2;
            end
        end
    end
    %% Summaries
    results = struct();
    results.alpha.draws = alpha_draws;
    results.alpha.mean  = mean(alpha_draws);
    results.alpha.std   = std(alpha_draws);
    results.alpha.quantiles = quantile(alpha_draws, [0.025,0.05,0.25,0.5,0.75,0.95,0.975]);

    results.kappa.draws = kappa_draws;
    results.kappa.mean  = mean(kappa_draws);
    results.kappa.std   = std(kappa_draws);
    results.kappa.quantiles     = quantile(kappa_draws, [0.025,0.05,0.25,0.5,0.75,0.95,0.975]);

    results.sigma_v2.draws = sigma_draws;
    results.sigma_v2.mean  = mean(sigma_draws);
    results.sigma_v2.std   = std(sigma_draws);
    results.sigma_v2.quantiles = quantile(sigma_draws, [0.025,0.05,0.25,0.5,0.75,0.95,0.975]);

    results.priors = struct('mu_alpha',mu_alpha,'sigma_alpha',sigma_alpha, ...
                            'mu_kappa',mu_kappa,'sigma_kappa',sigma_kappa, ...
                            'a_sig',a_sig,'b_sig',b_sig);
    results.opts   = struct('alpha0',alpha,'kappa0',kappa,'sigma_v20',sigma_v2, ...
                            'seed',seed,'constrain_alpha',constrain_alpha, ...
                            'store_every',store_every);
    if verbose
        fprintf('\n=== NKPC (Gibbs) — Posterior (means and 95%% CI) ===\n');
        fprintf('alpha:  mean %.4f  [%.4f, %.4f]\n', results.alpha.mean, results.alpha.quantiles(1), results.alpha.quantiles(7));
        fprintf('kappa:  mean %.4f  [%.4f, %.4f]\n', results.kappa.mean, results.kappa.quantiles(1), results.kappa.quantiles(7));
        fprintf('sigma2: mean %.4f  [%.4f, %.4f]\n', results.sigma_v2.mean, results.sigma_v2.quantiles(1), results.sigma_v2.quantiles(7));
    end
end

function v = getfield_with_default(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end