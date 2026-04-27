function results = func_nkpc_hsa_tv_xerror(pi_data, pi_prev_data, Epi_data, x_data, ...
                                        Nhat_data, Nbar_data, ...
                                        n_burn, n_keep, priors, opts)
% Estimates NKPC HSA with DETERMINISTIC time-varying kappa_t AND Measurement Error in x
%
% Model:
%   pi_t = alpha*pi_{t-1} + (1-alpha)E_t pi_{t+1} + kappa_t * x_t^* - theta Nhat_t + eps_t
%   x_t^{obs} = x_t^* + u_t
%   u_t ~ N(0, sigma_u^2)
%   kappa_t = kappa_0 + delta * (Nbar_t - Nbar_0)

    %% ---------------- Data ----------------
    T = numel(pi_data);
    pi_t     = pi_data(:);
    pi_tm1   = pi_prev_data(:);
    E_pi_tp1 = Epi_data(:);
    x_obs    = x_data(:);       % <--- Renamed to x_obs
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
    
    % Variance hyper for NKPC (sigma_v2)
    a_v         = getd(priors,'a_v',         2.0);
    b_v         = getd(priors,'b_v',         2.0);
    
    % [NEW] Variance hyper for Measurement Error (sigma_u2)
    a_u         = getd(priors,'a_u',         2.0);
    b_u         = getd(priors,'b_u',         2.0);

    %% ---------------- Options ----------------
    if nargin < 10 || isempty(opts), opts = struct(); end
    alpha       = getd(opts,'alpha0',      0.6);
    theta       = getd(opts,'theta0',      0.5);
    kappa0      = getd(opts,'kappa00',     0.3);
    delta       = getd(opts,'delta0',      0.0);
    sigma_v2    = getd(opts,'sigma_v20',   1.0);
    sigma_u2    = getd(opts,'sigma_u20',   0.1); % [NEW] Initial ME variance
    
    seed        = getd(opts,'seed',        []);
    verbose     = getd(opts,'verbose',     true);
    store_every = max(1, getd(opts,'store_every', 1));
    
    if ~isempty(seed), rng(seed); end

    %% ---------------- Initialization ----------------
    % Initialize latent true x (x_star) with observed data
    x_star = x_obs; 

    %% ---------------- Storage ----------------
    n_store = ceil(n_keep/store_every);
    alpha_draws  = zeros(n_store,1);
    theta_draws  = zeros(n_store,1);
    kappa0_draws = zeros(n_store,1);
    delta_draws  = zeros(n_store,1);
    sv_draws     = zeros(n_store,1);
    su_draws     = zeros(n_store,1); % [NEW]
    kappa_draws  = zeros(n_store,T);
    x_star_mean  = zeros(T,1);       % To store posterior mean of latent x

    if verbose
        fprintf('Gibbs start: burn-in=%d, keep=%d (TV Kappa + ME on x)\n', n_burn, n_keep);
    end

    %% ---------------- Gibbs loop ----------------
    total_iter = n_burn + n_keep;
    store_idx = 0;
    
    for iter = 1:total_iter
        
        % Compute current kappa_t
        kappa_t = kappa0 + delta * dNbar_cum;
        
        % ===== 1. alpha | . =====
        % Use x_star instead of x_obs
        y_a = pi_t - E_pi_tp1 - kappa_t.*x_star + theta.*Nhat; 
        X_a = pi_tm1 - E_pi_tp1;
        
        prec0 = 1/(sigma_alpha^2);
        precD = (X_a'*X_a)/sigma_v2;
        postV = 1/(prec0 + precD);
        postM = postV*(prec0*mu_alpha + (X_a'*y_a)/sigma_v2);
        alpha = postM + sqrt(postV)*randn;
        
        % ===== 2. theta | . =====
        y_th = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 - kappa_t.*x_star;
        X_th = -Nhat;
        
        prec0 = 1/(sigma_theta^2);
        precD = (X_th'*X_th)/sigma_v2;
        postV = 1/(prec0 + precD);
        postM = postV*(prec0*mu_theta + (X_th'*y_th)/sigma_v2);
        theta = postM + sqrt(postV)*randn;
        
        % ===== 3. kappa0, delta | . =====
        resid = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 + theta.*Nhat;
        
        % Regressor now uses x_star
        X_kappa = [x_star, dNbar_cum.*x_star]; 
        
        Sigma0_inv = diag([1/sigma_kappa0^2, 1/sigma_delta^2]);
        mu0 = [mu_kappa0; mu_delta];
        
        Sigma_post_inv = Sigma0_inv + (X_kappa'*X_kappa)/sigma_v2;
        Sigma_post = inv(Sigma_post_inv);
        mu_post = Sigma_post * (Sigma0_inv*mu0 + (X_kappa'*resid)/sigma_v2);
        
        params = mu_post + chol(Sigma_post,'lower')*randn(2,1);
        kappa0 = params(1);
        delta  = params(2);
        
        % Update kappa_t with new params
        kappa_t = kappa0 + delta * dNbar_cum;
        
        % ===== 4. sigma_v2 (NKPC error) | . =====
        eps = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 - kappa_t.*x_star + theta.*Nhat;
        
        a_post = a_v + T/2;
        b_post = b_v + 0.5*sum(eps.^2);
        sigma_v2 = 1/gamrnd(a_post, 1/b_post);

        % ===== [NEW] 5. x_star (Latent true x) | . =====
        % We have two sources of info for x_star:
        % 1. Measurement: x_obs ~ N(x_star, sigma_u2)
        % 2. Structure:   pi_t ~ N( ... + kappa_t*x_star, sigma_v2)
        
        % Calculate residuals excluding x term from NKPC
        resid_pi = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 + theta.*Nhat;
        
        % Posterior Precision for each t
        % 1/sigma_post^2 = (kappa_t^2)/sigma_v2 + 1/sigma_u2
        prec_x = (kappa_t.^2)./sigma_v2 + 1/sigma_u2;
        var_x  = 1./prec_x;
        
        % Posterior Mean for each t
        % mean = var * [ (kappa_t * resid_pi)/sigma_v2 + x_obs/sigma_u2 ]
        mu_x = var_x .* ( (kappa_t .* resid_pi)./sigma_v2 + x_obs./sigma_u2 );
        
        % Draw x_star
        x_star = mu_x + sqrt(var_x) .* randn(T,1);
        
        % ===== [NEW] 6. sigma_u2 (Measurement Error) | . =====
        % u_t = x_obs - x_star
        u_err = x_obs - x_star;
        
        a_u_post = a_u + T/2;
        b_u_post = b_u + 0.5*sum(u_err.^2);
        sigma_u2 = 1/gamrnd(a_u_post, 1/b_u_post);

        % ===== 7. Store =====
        if iter > n_burn && mod(iter - n_burn, store_every)==0
            store_idx = store_idx + 1;
            alpha_draws(store_idx)  = alpha;
            theta_draws(store_idx)  = theta;
            kappa0_draws(store_idx) = kappa0;
            delta_draws(store_idx)  = delta;
            sv_draws(store_idx)     = sigma_v2;
            su_draws(store_idx)     = sigma_u2; % Store ME variance
            kappa_draws(store_idx,:) = kappa_t';
            
            % Accumulate x_star for averaging
            x_star_mean = x_star_mean + x_star;
        end
        
        if verbose && mod(iter, 2000)==0
            fprintf('Iter %d/%d: α=%.2f κ0=%.2f σv²=%.3f σu²=%.3f\n', ...
                    iter, total_iter, alpha, kappa0, sigma_v2, sigma_u2);
        end
    end
    
    % Average x_star
    x_star_mean = x_star_mean / n_store;

    %% ---------------- Results ----------------
    results = struct();
    results.alpha      = summarize(alpha_draws);
    results.theta      = summarize(theta_draws);
    results.kappa0     = summarize(kappa0_draws);
    results.delta      = summarize(delta_draws);
    results.sigma_v2   = summarize(sv_draws);
    results.sigma_u2   = summarize(su_draws); % New result
    
    results.states.kappa_mean = mean(kappa_draws,1)';
    results.states.kappa_draws = kappa_draws;
    results.states.x_star_mean = x_star_mean; % Latent x estimate
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