function results = func_nkpc_hsa(pi_data, pi_prev_data, Epi_data, x_data, Nhat_data, ...
                                 n_burn, n_keep, priors, opts)
% NKPC with observed cycle Nhat: y* = X beta + v, v~N(0, sigma_v^2)
% beta = [alpha; kappa; theta]. No FFBS (Nhat is given).
%
% priors (optional; σ = std):
%   mu_alpha(0.5),  sigma_alpha(0.2)
%   mu_kappa(0.0),  sigma_kappa(0.3)
%   mu_theta(0.0),  sigma_theta(0.3)
%   a_v(2.0), b_v(2.0)   % Inv-Gamma(shape, scale) for sigma_v^2
%
% opts (optional):
%   alpha0(0.6), kappa0(0.3), theta0(0.5), sigma_v20(1.0)
%   seed([]), verbose(true), store_every(1)
%   constrain_alpha(false)  % if true, reject draws outside (0,1)

    %% Data
    pi_t     = pi_data(:);
    pi_tm1   = pi_prev_data(:);
    E_pi_tp1 = Epi_data(:);
    x_t      = x_data(:);
    Nhat     = Nhat_data(:);
    T = numel(pi_t);
    if any([numel(pi_tm1), numel(E_pi_tp1), numel(x_t), numel(Nhat)] ~= T)
        error('All input series must have the same length T.');
    end
    if any(~isfinite([pi_t; pi_tm1; E_pi_tp1; x_t; Nhat]))
        error('Input data contain non-finite values.');
    end
    %% Priors with defaults
    if nargin < 8 || isempty(priors), priors = struct(); end
    mu_alpha    = getd(priors,'mu_alpha',    0.5);
    sigma_alpha = getd(priors,'sigma_alpha', 0.2);
    mu_kappa    = getd(priors,'mu_kappa',    0.0);
    sigma_kappa = getd(priors,'sigma_kappa', 0.3);
    mu_theta    = getd(priors,'mu_theta',    0.0);
    sigma_theta = getd(priors,'sigma_theta', 0.3);
    a_v         = getd(priors,'a_v',         2.0);
    b_v         = getd(priors,'b_v',         2.0);
    if any([sigma_alpha, sigma_kappa, sigma_theta, a_v, b_v] <= 0)
        error('Priors must be positive where applicable.');
    end
    %% Options with defaults
    if nargin < 9 || isempty(opts), opts = struct(); end
    alpha     = getd(opts,'alpha0',     0.6);
    kappa     = getd(opts,'kappa0',     0.3);
    theta     = getd(opts,'theta0',     0.5);
    sigma_v2  = getd(opts,'sigma_v20',  1.0);
    seed      = getd(opts,'seed',       []);
    verbose   = getd(opts,'verbose',    true);
    store_every = max(1, getd(opts,'store_every', 1));
    if ~isempty(seed), rng(seed); end
    %% Storage
    n_store = ceil(n_keep / store_every);
    alpha_draws = zeros(n_store,1);
    kappa_draws = zeros(n_store,1);
    theta_draws = zeros(n_store,1);
    sv_draws    = zeros(n_store,1);

    if verbose
        fprintf('Gibbs start: burn-in=%d, keep=%d (thin=%d)\n', n_burn, n_keep, store_every);
    end

    %% Gibbs
    total_iter = n_burn + n_keep;
    store_idx = 0;

    for iter = 1:total_iter
        % ---- NKPC: π_t = α π_{t-1} + (1−α)Eπ_{t+1} + κ x_t − θ N̂_t + v_t

        % alpha | .
        y_a = pi_t - E_pi_tp1 - kappa.*x_t + theta.*Nhat;
        X_a = pi_tm1 - E_pi_tp1;
        Sxx = X_a' * X_a;
        if Sxx > 1e-12
            prec0 = 1/(sigma_alpha^2);
            postP = prec0 + Sxx / sigma_v2;
            postV = 1 / postP;
            postM = postV * (prec0*mu_alpha + (X_a' * y_a)/sigma_v2);
            a_draw = postM + sqrt(postV) * randn;            
            alpha = a_draw;
        end

        % kappa | .
        y_k = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 + theta.*Nhat;
        X_k = x_t;
        Sxx = X_k' * X_k;
        if Sxx > 1e-12
            prec0 = 1/(sigma_kappa^2);
            postP = prec0 + Sxx / sigma_v2;
            postV = 1 / postP;
            postM = postV * (prec0*mu_kappa + (X_k' * y_k)/sigma_v2);
            kappa = postM + sqrt(postV) * randn;
        end

        % theta | .
        y_thet = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 - kappa.*x_t;
        X_thet = -Nhat; % note minus sign
        Sxx = X_thet' * X_thet;
        if Sxx > 1e-12
            prec0 = 1/(sigma_theta^2);
            postP = prec0 + Sxx / sigma_v2;
            postV = 1 / postP;
            postM = postV * (prec0*mu_theta + (X_thet' * y_thet)/sigma_v2);
            theta = postM + sqrt(postV) * randn;
        end

        % ---- sigma_v^2 | .
        resid = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 - kappa.*x_t + theta.*Nhat;
        a_post = a_v + T/2;
        b_post = b_v + 0.5 * sum(resid.^2);
        sigma_v2 = 1 / gamrnd(a_post, 1/b_post);

        % ---- Store
        if iter > n_burn && mod(iter - n_burn, store_every) == 0
            store_idx = store_idx + 1;
            alpha_draws(store_idx) = alpha;
            kappa_draws(store_idx) = kappa;
            theta_draws(store_idx) = theta;
            sv_draws(store_idx)    = sigma_v2;
        end      
    end

    %% Results
    summarize = @(v) struct('draws',v,'mean',mean(v),'std',std(v), ...
                            'quantiles',quantile(v,[0.025,0.05,0.25,0.5,0.75,0.95,0.975]));
    results.alpha    = summarize(alpha_draws);
    results.kappa    = summarize(kappa_draws);
    results.theta    = summarize(theta_draws);
    results.sigma_v2 = summarize(sv_draws);
    results.priors   = priors;
    results.opts     = opts;

    if verbose
        fprintf('\nPosterior (means and 95%% CI):\n');
        fprintf('alpha: %.4f  [%.4f, %.4f]\n', results.alpha.mean, results.alpha.quantiles(1), results.alpha.quantiles(7));
        fprintf('kappa: %.4f  [%.4f, %.4f]\n', results.kappa.mean, results.kappa.quantiles(1), results.kappa.quantiles(7));
        fprintf('theta: %.4f  [%.4f, %.4f]\n', results.theta.mean, results.theta.quantiles(1), results.theta.quantiles(7));
    end
end

%% ----------- Helpers -----------
function val = getd(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), val = s.(f); else, val = d; end
end