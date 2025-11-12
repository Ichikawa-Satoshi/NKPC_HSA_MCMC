function results = func_nkpc_ces_xerror(pi_data, pi_prev_data, Epi_data, x_data, n_burn, n_keep, priors, opts)
% Estimates NKPC CES parameters (alpha, kappa, sigma_v^2, sigma_u^2) via Gibbs sampling
% with i.i.d. measurement error in x: x_t = x*_t + u_t, u_t ~ N(0, sigma_u^2)
% Priors and options are supplied externally.

% priors fields (all optional, defaults in parentheses):
%   mu_alpha(0.5), sigma_alpha(0.2)
%   mu_kappa(0.0), sigma_kappa(0.5)
%   a_sig(2.0), b_sig(2.0)   % Inv-Gamma(shape=a_sig, scale=b_sig) for sigma_v^2
%   a_sigu(2.0), b_sigu(0.5) % Inv-Gamma(shape=a_sigu, scale=b_sigu) for sigma_u^2

% opts fields (all optional):
%   alpha0(0.5), kappa0(0.1), sigma_v20(0.1), sigma_u20(0.01)
%   seed([] -> no rng set), constrain_alpha(false), verbose(true), store_every(1)

    %% Data
    pi_t     = pi_data(:);
    pi_tm1   = pi_prev_data(:);
    E_pi_tp1 = Epi_data(:);
    x_obs    = x_data(:);  % observed x with measurement error
    T_obs = numel(pi_t);
    if any([numel(pi_tm1), numel(E_pi_tp1), numel(x_obs)] ~= T_obs)
        error('Input vectors must have the same length.');
    end
    %% Priors (with safe defaults)
    if nargin < 7 || isempty(priors), priors = struct(); end
    mu_alpha    = getfield_with_default(priors, 'mu_alpha',    0.5);
    sigma_alpha = getfield_with_default(priors, 'sigma_alpha', 0.2);
    mu_kappa    = getfield_with_default(priors, 'mu_kappa',    0.0);
    sigma_kappa = getfield_with_default(priors, 'sigma_kappa', 0.5);
    a_sig       = getfield_with_default(priors, 'a_sig',       2.0);
    b_sig       = getfield_with_default(priors, 'b_sig',       2.0);
    a_sigu      = getfield_with_default(priors, 'a_sigu',      2.0);
    b_sigu      = getfield_with_default(priors, 'b_sigu',      0.5);
    if sigma_alpha <= 0 || sigma_kappa <= 0 || a_sig <= 0 || b_sig <= 0 || a_sigu <= 0 || b_sigu <= 0
        error('Prior hyperparameters must be positive where applicable.');
    end
    %% Options
    if nargin < 8 || isempty(opts), opts = struct(); end
    alpha    = getfield_with_default(opts, 'alpha0',    0.5);
    kappa    = getfield_with_default(opts, 'kappa0',    0.1);
    sigma_v2 = getfield_with_default(opts, 'sigma_v20', 0.1);
    sigma_u2 = getfield_with_default(opts, 'sigma_u20', 0.01);
    seed     = getfield_with_default(opts, 'seed',      []);
    constrain_alpha = getfield_with_default(opts, 'constrain_alpha', false);
    verbose = getfield_with_default(opts, 'verbose', true);
    store_every = max(1, getfield_with_default(opts, 'store_every', 1));
    if ~isempty(seed), rng(seed); end
    % Initialize latent x* at observed values
    x_star = x_obs;
    if verbose
        fprintf('Initial: alpha=%.3f, kappa=%.3f, sigma_v2=%.3f, sigma_u2=%.4f\n', ...
                alpha, kappa, sigma_v2, sigma_u2);
        fprintf('Burn-in: %d, Keep: %d (store every %d)\n', n_burn, n_keep, store_every);
    end
    %% Storage
    n_store = ceil(n_keep / store_every);
    alpha_draws   = zeros(n_store,1);
    kappa_draws   = zeros(n_store,1);
    sigma_v_draws = zeros(n_store,1);
    sigma_u_draws = zeros(n_store,1);
    x_star_draws  = zeros(n_store, T_obs);  % store latent x*
    store_idx = 0;
    %% Gibbs
    total_iter = n_burn + n_keep;
    for iter = 1:total_iter
        % ---- Sample x*_t | alpha, kappa, sigma_v2, sigma_u2, data (for each t)
        % Posterior for x*_t is normal:
        % From NKPC: π_t - α π_{t-1} - (1-α)E[π_{t+1}] = κ x*_t + v_t
        % From measurement: x_obs_t = x*_t + u_t
        for t = 1:T_obs
            % Precision from measurement equation
            prec_u = 1/sigma_u2;
            % Precision from structural equation
            prec_v = kappa^2 / sigma_v2;
            % Posterior precision and variance
            post_prec_x = prec_u + prec_v;
            post_var_x = 1 / post_prec_x;          
            % Posterior mean
            y_resid = pi_t(t) - alpha * pi_tm1(t) - (1 - alpha) * E_pi_tp1(t);
            post_mean_x = post_var_x * (prec_u * x_obs(t) + (kappa / sigma_v2) * y_resid);            
            x_star(t) = post_mean_x + sqrt(post_var_x) * randn;
        end

        % ---- Sample alpha | kappa, sigma_v2, x*, data
        y_alpha = pi_t - E_pi_tp1 - kappa .* x_star;
        X_alpha = pi_tm1 - E_pi_tp1;
        prior_prec_a = 1/(sigma_alpha^2);
        data_prec_a  = (X_alpha' * X_alpha) / sigma_v2;
        post_prec_a  = prior_prec_a + data_prec_a;
        post_var_a   = 1 / post_prec_a;
        post_mean_a  = post_var_a * (prior_prec_a * mu_alpha + (X_alpha' * y_alpha) / sigma_v2);
        alpha_draw = post_mean_a + sqrt(post_var_a) * randn;
        if constrain_alpha
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
        % ---- Sample kappa | alpha, sigma_v2, x*, data
        y_kappa = pi_t - alpha .* pi_tm1 - (1 - alpha) .* E_pi_tp1;
        X_kappa = x_star;
        prior_prec_k = 1/(sigma_kappa^2);
        data_prec_k  = (X_kappa' * X_kappa) / sigma_v2;
        post_prec_k  = prior_prec_k + data_prec_k;
        post_var_k   = 1 / post_prec_k;
        post_mean_k  = post_var_k * (prior_prec_k * mu_kappa + (X_kappa' * y_kappa) / sigma_v2);
        kappa = post_mean_k + sqrt(post_var_k) * randn;
        % ---- Sample sigma_v^2 | alpha, kappa, x*, data
        resid = pi_t - alpha .* pi_tm1 - (1 - alpha) .* E_pi_tp1 - kappa .* x_star;
        a_post_v = a_sig + T_obs/2;
        b_post_v = b_sig + 0.5 * sum(resid.^2);
        sigma_v2 = 1 / gamrnd(a_post_v, 1/b_post_v);
        % ---- Sample sigma_u^2 | x*, data
        resid_u = x_obs - x_star;
        a_post_u = a_sigu + T_obs/2;
        b_post_u = b_sigu + 0.5 * sum(resid_u.^2);
        sigma_u2 = 1 / gamrnd(a_post_u, 1/b_post_u);
        % ---- Store after burn-in (with thinning)
        if iter > n_burn
            if mod(iter - n_burn, store_every) == 0
                store_idx = store_idx + 1;
                alpha_draws(store_idx)   = alpha;
                kappa_draws(store_idx)   = kappa;
                sigma_v_draws(store_idx) = sigma_v2;
                sigma_u_draws(store_idx) = sigma_u2;
                x_star_draws(store_idx, :) = x_star';
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
    results.kappa.quantiles = quantile(kappa_draws, [0.025,0.05,0.25,0.5,0.75,0.95,0.975]);

    results.sigma_v2.draws = sigma_v_draws;
    results.sigma_v2.mean  = mean(sigma_v_draws);
    results.sigma_v2.std   = std(sigma_v_draws);
    results.sigma_v2.quantiles = quantile(sigma_v_draws, [0.025,0.05,0.25,0.5,0.75,0.95,0.975]);

    results.sigma_u2.draws = sigma_u_draws;
    results.sigma_u2.mean  = mean(sigma_u_draws);
    results.sigma_u2.std   = std(sigma_u_draws);
    results.sigma_u2.quantiles = quantile(sigma_u_draws, [0.025,0.05,0.25,0.5,0.75,0.95,0.975]);

    results.x_star.draws = x_star_draws;
    results.x_star.mean  = mean(x_star_draws, 1)';
    results.x_star.std   = std(x_star_draws, 0, 1)';
    results.x_star.quantiles = quantile(x_star_draws, [0.025,0.5,0.975], 1)';

    results.priors = struct('mu_alpha',mu_alpha,'sigma_alpha',sigma_alpha, ...
                            'mu_kappa',mu_kappa,'sigma_kappa',sigma_kappa, ...
                            'a_sig',a_sig,'b_sig',b_sig, ...
                            'a_sigu',a_sigu,'b_sigu',b_sigu);
    results.opts   = struct('alpha0',alpha,'kappa0',kappa,'sigma_v20',sigma_v2, ...
                            'sigma_u20',sigma_u2,'seed',seed, ...
                            'constrain_alpha',constrain_alpha,'store_every',store_every);
    
    if verbose
        fprintf('\n=== NKPC with X Measurement Error (Gibbs) — Posterior (means and 95%% CI) ===\n');
        fprintf('alpha:    mean %.4f  [%.4f, %.4f]\n', results.alpha.mean, results.alpha.quantiles(1), results.alpha.quantiles(7));
        fprintf('kappa:    mean %.4f  [%.4f, %.4f]\n', results.kappa.mean, results.kappa.quantiles(1), results.kappa.quantiles(7));
        fprintf('sigma_v2: mean %.4f  [%.4f, %.4f]\n', results.sigma_v2.mean, results.sigma_v2.quantiles(1), results.sigma_v2.quantiles(7));
        fprintf('sigma_u2: mean %.4f  [%.4f, %.4f]\n', results.sigma_u2.mean, results.sigma_u2.quantiles(1), results.sigma_u2.quantiles(7));
    end
end

function v = getfield_with_default(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end