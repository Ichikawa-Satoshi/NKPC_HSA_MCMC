function results = func_nkpc_hsa_tv(pi_data, pi_prev_data, Epi_data, x_data, ...
                                        Nhat_data, Nbar_data, ...
                                        n_burn, n_keep, priors, opts)
% Estimates NKPC HSA with DETERMINISTIC time-varying kappa_t via Gibbs sampler
%
% pi_t = alpha*pi_{t-1} + (1-alpha)E_t pi_{t+1} + kappa_t x_t - theta Nhat_t + eps_t
% kappa_t = kappa_0 + delta * (Nbar_t - Nbar_0)   [DETERMINISTIC]

    %% ---------------- Data ----------------
    T = numel(pi_data);
    pi_t     = pi_data(:);
    pi_tm1   = pi_prev_data(:);
    E_pi_tp1 = Epi_data(:);
    x_t      = x_data(:);
    Nhat     = Nhat_data(:);
    Nbar     = Nbar_data(:);

    % Cumulative change from initial: (N̄_t - N̄_0)
    dNbar_cum = Nbar - Nbar(1);

    %% ---------------- Priors ----------------
    if nargin < 9 || isempty(priors), priors = struct(); end
    mu_alpha    = getd(priors,'mu_alpha',    0.5);
    sigma_alpha = getd(priors,'sigma_alpha', 0.2);
    mu_theta    = getd(priors,'mu_theta',    0.0);
    sigma_theta = getd(priors,'sigma_theta', 0.3);
    mu_kappa0   = getd(priors,'mu_kappa0',   0.3);
    sigma_kappa0= getd(priors,'sigma_kappa0',0.2);
    mu_delta    = getd(priors,'mu_delta',    0.0);
    sigma_delta = getd(priors,'sigma_delta', 0.3);

    % variance hyper
    a_v         = getd(priors,'a_v',         2.0);
    b_v         = getd(priors,'b_v',         2.0);

    %% ---------------- Options ----------------
    if nargin < 10 || isempty(opts), opts = struct(); end
    alpha       = getd(opts,'alpha0',      0.6);
    theta       = getd(opts,'theta0',      0.5);
    kappa0      = getd(opts,'kappa00',     0.3);
    delta       = getd(opts,'delta0',      0.0);
    sigma_v2    = getd(opts,'sigma_v20',   1.0);
    seed        = getd(opts,'seed',        []);
    verbose     = getd(opts,'verbose',     true);
    store_every = max(1, getd(opts,'store_every', 1));
    
    if ~isempty(seed), rng(seed); end

    %% ---------------- Storage ----------------
    n_store = ceil(n_keep/store_every);
    alpha_draws  = zeros(n_store,1);
    theta_draws  = zeros(n_store,1);
    kappa0_draws = zeros(n_store,1);
    delta_draws  = zeros(n_store,1);
    sv_draws     = zeros(n_store,1);
    kappa_draws  = zeros(n_store,T);

    if verbose
        fprintf('Gibbs start: burn-in=%d, keep=%d (DETERMINISTIC kappa)\n', n_burn, n_keep);
    end

    %% ---------------- Gibbs loop ----------------
    total_iter = n_burn + n_keep;
    store_idx = 0;
    
    for iter = 1:total_iter
        
        % Compute current kappa_t = kappa_0 + delta * (Nbar_t - Nbar_0)
        kappa_t = kappa0 + delta * dNbar_cum;
        
        % ===== 1. alpha | . =====
        % y = pi_t - E_t pi_{t+1} + kappa_t x_t - theta Nhat_t
        y_a = pi_t - E_pi_tp1 + kappa_t.*x_t - theta.*Nhat;
        X_a = pi_tm1 - E_pi_tp1;  % (pi_{t-1} - E_t pi_{t+1})
        
        prec0 = 1/(sigma_alpha^2);
        precD = (X_a'*X_a)/sigma_v2;
        postV = 1/(prec0 + precD);
        postM = postV*(prec0*mu_alpha + (X_a'*y_a)/sigma_v2);
        alpha = postM + sqrt(postV)*randn;
        
        % ===== 2. theta | . =====
        y_th = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 + kappa_t.*x_t;
        X_th = -Nhat;
        
        prec0 = 1/(sigma_theta^2);
        precD = (X_th'*X_th)/sigma_v2;
        postV = 1/(prec0 + precD);
        postM = postV*(prec0*mu_theta + (X_th'*y_th)/sigma_v2);
        theta = postM + sqrt(postV)*randn;
        
        % ===== 3. kappa0, delta | . =====
        % Residual: r_t = pi_t - alpha*pi_{t-1} - (1-alpha)E_t pi_{t+1} + theta*Nhat_t
        resid = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 + theta.*Nhat;
        
        % Linear regression: resid = kappa_t * x_t + noise
        %                           = (kappa0 + delta*dNbar_cum) * x_t + noise
        %                           = kappa0*x_t + delta*(dNbar_cum.*x_t) + noise
        
        X_kappa = [x_t, dNbar_cum.*x_t];  % T x 2 design matrix
        
        % Prior: [kappa0; delta] ~ N([mu_kappa0; mu_delta], diag([sigma_kappa0^2; sigma_delta^2]))
        Sigma0_inv = diag([1/sigma_kappa0^2, 1/sigma_delta^2]);
        mu0 = [mu_kappa0; mu_delta];
        
        % Posterior
        Sigma_post_inv = Sigma0_inv + (X_kappa'*X_kappa)/sigma_v2;
        Sigma_post = inv(Sigma_post_inv);
        mu_post = Sigma_post * (Sigma0_inv*mu0 + (X_kappa'*resid)/sigma_v2);
        
        params = mu_post + chol(Sigma_post,'lower')*randn(2,1);
        kappa0 = params(1);
        delta  = params(2);
        
        % ===== 4. sigma_v2 | . =====
        kappa_t = kappa0 + delta * dNbar_cum;
        eps = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 + kappa_t.*x_t - theta.*Nhat;
        
        a_post = a_v + T/2;
        b_post = b_v + 0.5*sum(eps.^2);
        sigma_v2 = 1/gamrnd(a_post, 1/b_post);
        
        % ===== 5. Store =====
        if iter > n_burn && mod(iter - n_burn, store_every)==0
            store_idx = store_idx + 1;
            alpha_draws(store_idx)  = alpha;
            theta_draws(store_idx)  = theta;
            kappa0_draws(store_idx) = kappa0;
            delta_draws(store_idx)  = delta;
            sv_draws(store_idx)     = sigma_v2;
            kappa_draws(store_idx,:) = kappa_t';
        end

        if verbose && mod(iter, 2000)==0
            fprintf('Iter %d/%d: α=%.3f θ=%.3f κ0=%.3f δ=%.4f σv²=%.3f\n', ...
                    iter, total_iter, alpha, theta, kappa0, delta, sigma_v2);
        end
    end

    %% ---------------- Results ----------------
    results = struct();
    results.alpha      = summarize(alpha_draws);
    results.theta      = summarize(theta_draws);
    results.kappa0     = summarize(kappa0_draws);
    results.delta      = summarize(delta_draws);
    results.sigma_v2   = summarize(sv_draws);
    results.states.kappa_mean = mean(kappa_draws,1)';   % T×1
    results.states.kappa_draws = kappa_draws;            % n_store × T
end

%% ====== Helpers =====================================================
function S = summarize(v)
    S.draws     = v;
    S.mean      = mean(v);
    S.std       = std(v);
    S.quantiles = quantile(v,[0.025 0.05 0.25 0.5 0.75 0.95 0.975]);
end

function val = getd(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f))
        val = s.(f);
    else
        val = d;
    end
end