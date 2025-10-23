function results = func_nkpc_hsa_decomp_xerror(pi_data, pi_prev_data, Epi_data, x_data, N_data, n_burn, n_keep, priors, opts)
% Estimates NKPC HSA via state-space model with FFBS Gibbs sampler
% ** NEW: Allows measurement error in x **
%
% ---- Externalized priors & options ----
% priors (all optional; defaults in parentheses, σ=std):
%   mu_alpha(0.5),  sigma_alpha(0.2)
%   mu_kappa(0.0),  sigma_kappa(0.3)
%   mu_theta(0.0),  sigma_theta(0.3)
%   mu_rho1(0.5),   sigma_rho1(0.2)
%   mu_rho2(-0.5),  sigma_rho2(0.2)
%   mu_n(0.0),      sigma_n(0.1)
%   a_v(2.0),   b_v(2.0)        % InvGamma(shape, scale) for sigma_v^2
%   a_eps(2.0), b_eps(2.0)      % for sigma_eps^2 (AR(2))
%   a_eta(2.0), b_eta(2.0)      % for sigma_eta^2 (RW trend)
%   ** NEW priors for x measurement error **
%   mu_x_star(0.0),    sigma_x_star(1.0)   % prior for x*_t (i.i.d.)
%   a_x_obs(2.0),      b_x_obs(0.05)       % for sigma_x_obs^2
%
% opts (all optional):
%   alpha0(0.6), kappa0(0.3), theta0(0.5), rho10(0.5), rho20(-0.5), n0(0.01)
%   sigma_v20(1.0), sigma_eps20(0.5), sigma_eta20(0.1)
%   ** NEW **
%   sigma_x_obs20(0.1)  % initial value for x observation error variance
%   seed([]), verbose(true), store_every(1)
%   constrain_alpha(false)  % if true, reject draws outside (0,1)
%   enforce_stationary(true) % AR(2) stationarity via rejection
%   r_target_scale(0.1)     % obs noise for target y in AR(2) FF
%   r_rw_scale(0.1)         % obs noise for RW FF
%
% Notes:
% - x_t = x*_t + ε^x_t, where x*_t ~ i.i.d. N(μ_x*, σ²_x*) and ε^x_t ~ N(0, σ²_x_obs)
% - NKPC uses true x*_t: π_t = α π_{t-1} + (1-α)Eπ_{t+1} + κ x*_t - θ N̂_t + v_t

    %% Data
    T = numel(pi_data);
    pi_t     = pi_data(:);
    pi_tm1   = pi_prev_data(:);
    E_pi_tp1 = Epi_data(:);
    x_t      = x_data(:);
    N_obs    = N_data(:);

    if any([numel(pi_tm1), numel(E_pi_tp1), numel(x_t), numel(N_obs)] ~= T)
        error('All input series must have the same length T.');
    end

    %% Priors with defaults
    if nargin < 8 || isempty(priors), priors = struct(); end
    mu_alpha    = getd(priors,'mu_alpha',    0.5);
    sigma_alpha = getd(priors,'sigma_alpha', 0.2);
    mu_kappa    = getd(priors,'mu_kappa',    0.0);
    sigma_kappa = getd(priors,'sigma_kappa', 0.3);
    mu_theta    = getd(priors,'mu_theta',    0.0);
    sigma_theta = getd(priors,'sigma_theta', 0.3);

    mu_rho1     = getd(priors,'mu_rho1',     0.5);
    sigma_rho1  = getd(priors,'sigma_rho1',  0.2);
    mu_rho2     = getd(priors,'mu_rho2',    -0.5);
    sigma_rho2  = getd(priors,'sigma_rho2',  0.2);

    mu_n        = getd(priors,'mu_n',        0.0);
    sigma_n     = getd(priors,'sigma_n',     0.1);

    a_v         = getd(priors,'a_v',         2.0);
    b_v         = getd(priors,'b_v',         2.0);
    a_eps       = getd(priors,'a_eps',       2.0);
    b_eps       = getd(priors,'b_eps',       2.0);
    a_eta       = getd(priors,'a_eta',       2.0);
    b_eta       = getd(priors,'b_eta',       2.0);

    % ** NEW: x measurement error priors **
    mu_x_star      = getd(priors,'mu_x_star',    0.0);
    sigma_x_star   = getd(priors,'sigma_x_star', 1.0);
    a_x_obs        = getd(priors,'a_x_obs',      2.0);
    b_x_obs        = getd(priors,'b_x_obs',      0.05);

    assertallpos([sigma_alpha,sigma_kappa,sigma_theta,sigma_rho1,sigma_rho2,sigma_n, ...
                  a_v,b_v,a_eps,b_eps,a_eta,b_eta,sigma_x_star,a_x_obs,b_x_obs], ...
        'Prior stds and IG params must be positive.');

    %% Options with defaults
    if nargin < 9 || isempty(opts), opts = struct(); end
    alpha      = getd(opts,'alpha0',     0.6);
    kappa      = getd(opts,'kappa0',     0.3);
    theta      = getd(opts,'theta0',     0.5);
    rho1       = getd(opts,'rho10',      0.5);
    rho2       = getd(opts,'rho20',     -0.5);
    n_drift    = getd(opts,'n0',         0.01);
    sigma_v2   = getd(opts,'sigma_v20',  1.0);
    sigma_eps2 = getd(opts,'sigma_eps20',0.5);
    sigma_eta2 = getd(opts,'sigma_eta20',0.1);
    
    % ** NEW **
    sigma_x_obs2 = getd(opts,'sigma_x_obs20', 0.1);

    seed      = getd(opts,'seed',       []);
    verbose   = getd(opts,'verbose',    true);
    store_every = max(1, getd(opts,'store_every', 1));

    constrain_alpha   = getd(opts,'constrain_alpha', false);
    enforce_station   = getd(opts,'enforce_stationary', true);
    r_target_scale    = getd(opts,'r_target_scale', 0.1);
    r_rw_scale        = getd(opts,'r_rw_scale',     0.1);

    if ~isempty(seed), rng(seed); end

    %% Initialize states
    % Initialize x_star as observed x_t
    x_star = x_t;
    
    % Initialize N decomposition
    Nbar = zeros(T,1);
    Nbar(1:min(2,T)) = N_obs(1:min(2,T));
    for t=3:T, Nbar(t) = 0.7*Nbar(t-1) + 0.3*N_obs(t); end
    Nhat = N_obs - Nbar;

    %% Storage
    n_store = ceil(n_keep/store_every);
    alpha_draws = zeros(n_store,1);
    kappa_draws = zeros(n_store,1);
    theta_draws = zeros(n_store,1);
    rho1_draws  = zeros(n_store,1);
    rho2_draws  = zeros(n_store,1);
    n_draws     = zeros(n_store,1);
    sv_draws    = zeros(n_store,1);
    se_draws    = zeros(n_store,1);
    seta_draws  = zeros(n_store,1);
    sx_obs_draws= zeros(n_store,1);  % NEW

    % Thinned state storage (diagnostics)
    Nbar_draws  = zeros(n_store, T);
    Nhat_draws  = zeros(n_store, T);
    x_star_draws= zeros(n_store, T);  % NEW

    if verbose
        fprintf('Gibbs start: burn-in=%d, keep=%d (thin=%d)\n', n_burn, n_keep, store_every);
        fprintf('** Model includes x measurement error **\n');
    end

    %% Gibbs
    total_iter = n_burn + n_keep;
    store_idx = 0;

    for iter = 1:total_iter
        % ---- NKPC: π_t = α π_{t-1} + (1−α)Eπ_{t+1} + κ x*_t − θ N̂_t + v_t
        % alpha | .
        y_a = pi_t - E_pi_tp1 - kappa.*x_star + theta.*Nhat;
        X_a = pi_tm1 - E_pi_tp1;
        if (X_a'*X_a) > 1e-12
            prec0 = 1/(sigma_alpha^2);
            precD = (X_a'*X_a)/sigma_v2;
            postP = prec0 + precD; postV = 1/postP;
            postM = postV*(prec0*mu_alpha + (X_a'*y_a)/sigma_v2);
            a_draw = postM + sqrt(postV)*randn;            
            alpha = a_draw;
        end

        % kappa | . (uses x_star, not x_t)
        y_k = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 + theta.*Nhat;
        X_k = x_star;  % ** IMPORTANT: use latent x_star **
        if (X_k'*X_k) > 1e-12
            prec0 = 1/(sigma_kappa^2);
            precD = (X_k'*X_k)/sigma_v2;
            postP = prec0 + precD; postV = 1/postP;
            postM = postV*(prec0*mu_kappa + (X_k'*y_k)/sigma_v2);
            kappa = postM + sqrt(postV)*randn;
        end

        % theta | .
        y_thet = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 - kappa.*x_star;
        X_thet = -Nhat; % note minus sign
        if (X_thet'*X_thet) > 1e-12
            prec0 = 1/(sigma_theta^2);
            precD = (X_thet'*X_thet)/sigma_v2;
            postP = prec0 + precD; postV = 1/postP;
            postM = postV*(prec0*mu_theta + (X_thet'*y_thet)/sigma_v2);
            theta = postM + sqrt(postV)*randn;
        end

        % ---- Sample x*_t (i.i.d. latent values)
        % x*_t has three sources of information:
        %   1. Prior: x*_t ~ N(μ_x*, σ²_x*)
        %   2. NKPC:  π_t ≈ ... + κ x*_t + ... with noise σ²_v
        %   3. Observation: x_t = x*_t + ε with noise σ²_x_obs
        for t = 1:T
            % NKPC-implied mean for x*_t
            nkpc_mean = (pi_t(t) - alpha*pi_tm1(t) - (1-alpha)*E_pi_tp1(t) + theta*Nhat(t)) / kappa;
            
            % Precisions
            prec_prior = 1/(sigma_x_star^2);
            prec_nkpc  = (kappa^2)/sigma_v2;
            prec_obs   = 1/sigma_x_obs2;
            
            % Posterior
            post_prec = prec_prior + prec_nkpc + prec_obs;
            post_var  = 1/post_prec;
            post_mean = post_var * (prec_prior*mu_x_star + prec_nkpc*nkpc_mean + prec_obs*x_t(t));
            
            x_star(t) = post_mean + sqrt(post_var)*randn;
        end

        % ---- AR(2) params for Nhat: N̂_t = ρ1 N̂_{t-1} + ρ2 N̂_{t-2} + ε_t
        if T>=3
            y_r = Nhat(3:end);
            X_r = [Nhat(2:end-1), Nhat(1:end-2)];
            if size(X_r,1)>0
                Prec0 = diag([1/(sigma_rho1^2), 1/(sigma_rho2^2)]);
                PrecD = (X_r'*X_r)/sigma_eps2;
                PostP = Prec0 + PrecD;
                PostC = inv(PostP);
                mu0   = [mu_rho1; mu_rho2];
                PostM = PostC*(Prec0*mu0 + (X_r'*y_r)/sigma_eps2);

                ok=false; tries=0; maxTries=2000;
                while ~ok && tries<maxTries
                    rdraw = mvnrnd(PostM, PostC)';
                    if ~enforce_station || is_stationary_ar2(rdraw(1), rdraw(2))
                        rho1 = rdraw(1); rho2 = rdraw(2); ok=true;
                    end
                    tries = tries+1;
                end
                if ~ok
                    % fallback to mean (mild shrink)
                    rho1 = PostM(1); rho2 = PostM(2);
                end
            end
        end

        % ---- Random walk drift n: N̄_t = n + N̄_{t-1} + η_t
        if T>=2
            y_n = Nbar(2:end) - Nbar(1:end-1);
            Tn = numel(y_n);
            prec0 = 1/(sigma_n^2);
            precD = Tn/sigma_eta2;
            postP = prec0 + precD; postV = 1/postP;
            postM = postV*(prec0*mu_n + sum(y_n)/sigma_eta2);
            n_drift = postM + sqrt(postV)*randn;
        end

        % ---- Variances
        % NKPC residual (use x_star)
        nkpc_resid = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 - kappa.*x_star + theta.*Nhat;
        a_post = a_v + T/2;
        b_post = b_v + 0.5*sum(nkpc_resid.^2);
        sigma_v2 = 1/gamrnd(a_post, 1/b_post);

        % AR(2) innovation variance
        if T>=3
            ar_res = Nhat(3:end) - rho1.*Nhat(2:end-1) - rho2.*Nhat(1:end-2);
            a_post = a_eps + numel(ar_res)/2;
            b_post = b_eps + 0.5*sum(ar_res.^2);
            sigma_eps2 = 1/gamrnd(a_post, 1/b_post);
        end

        % RW innovation variance
        if T>=2
            rw_res = Nbar(2:end) - n_drift - Nbar(1:end-1);
            a_post = a_eta + numel(rw_res)/2;
            b_post = b_eta + 0.5*sum(rw_res.^2);
            sigma_eta2 = 1/gamrnd(a_post, 1/b_post);
        end

        % ** NEW: x observation error variance **
        x_resid = x_t - x_star;
        a_post = a_x_obs + T/2;
        b_post = b_x_obs + 0.5*sum(x_resid.^2);
        sigma_x_obs2 = 1/gamrnd(a_post, 1/b_post);

        % ---- FFBS for states
        Nhat = sample_ar2_states_ffbs_ext(N_obs - Nbar, rho1, rho2, sigma_eps2, ...
                    pi_t, alpha, pi_tm1, E_pi_tp1, kappa, x_star, theta, sigma_v2, r_target_scale);
        Nbar = sample_rw_states_ffbs_ext(N_obs - Nhat, n_drift, sigma_eta2, r_rw_scale);

        % ---- Store
        if iter > n_burn && mod(iter - n_burn, store_every)==0
            store_idx = store_idx + 1;
            alpha_draws(store_idx)  = alpha;
            kappa_draws(store_idx)  = kappa;
            theta_draws(store_idx)  = theta;
            rho1_draws(store_idx)   = rho1;
            rho2_draws(store_idx)   = rho2;
            n_draws(store_idx)      = n_drift;
            sv_draws(store_idx)     = sigma_v2;
            se_draws(store_idx)     = sigma_eps2;
            seta_draws(store_idx)   = sigma_eta2;
            sx_obs_draws(store_idx) = sigma_x_obs2;  % NEW

            Nbar_draws(store_idx,:)   = Nbar';
            Nhat_draws(store_idx,:)   = Nhat';
            x_star_draws(store_idx,:) = x_star';  % NEW
        end

        if verbose && mod(iter, 5000)==0
            fprintf('Iter %d/%d: a=%.3f k=%.3f th=%.3f r1=%.3f r2=%.3f sx=%.4f\n', ...
                iter, total_iter, alpha, kappa, theta, rho1, rho2, sqrt(sigma_x_obs2));
        end
    end

    %% Results
    results = struct();
    add_sum = @(v) struct('draws',v,'mean',mean(v),'std',std(v), ...
                          'quantiles',quantile(v,[0.025,0.05,0.25,0.5,0.75,0.95,0.975]));
    results.alpha      = add_sum(alpha_draws);
    results.kappa      = add_sum(kappa_draws);
    results.theta      = add_sum(theta_draws);
    results.rho1       = add_sum(rho1_draws);
    results.rho2       = add_sum(rho2_draws);
    results.n          = add_sum(n_draws);
    results.sigma_v2   = add_sum(sv_draws);
    results.sigma_eps2 = add_sum(se_draws);
    results.sigma_eta2 = add_sum(seta_draws);
    results.sigma_x_obs2 = add_sum(sx_obs_draws);  % NEW

    results.states.Nbar_mean   = mean(Nbar_draws,1)';
    results.states.Nhat_mean   = mean(Nhat_draws,1)';
    results.states.N_mean      = results.states.Nbar_mean + results.states.Nhat_mean;
    results.states.x_star_mean = mean(x_star_draws,1)';  % NEW
    results.states.x_star_draws= x_star_draws;           % NEW (full draws)

    results.priors = priors;
    results.opts   = opts;

    if verbose
        fprintf('\n=== NKPC-HSA with x measurement error — Posterior means and 95%% CI ===\n');
        fprintf('alpha:       %.4f  [%.4f, %.4f]\n', results.alpha.mean, results.alpha.quantiles(1), results.alpha.quantiles(7));
        fprintf('kappa:       %.4f  [%.4f, %.4f]\n', results.kappa.mean, results.kappa.quantiles(1), results.kappa.quantiles(7));
        fprintf('theta:       %.4f  [%.4f, %.4f]\n', results.theta.mean, results.theta.quantiles(1), results.theta.quantiles(7));
        fprintf('sigma_x_obs: %.4f  [%.4f, %.4f]\n', sqrt(results.sigma_x_obs2.mean), ...
                sqrt(results.sigma_x_obs2.quantiles(1)), sqrt(results.sigma_x_obs2.quantiles(7)));
    end
end

%% ----------- Helpers -----------
function val = getd(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), val = s.(f); else, val = d; end
end

function assertallpos(v, msg)
    if any(~isfinite(v) | v<=0), error(msg); end
end

function tf = is_stationary_ar2(r1, r2)
    % sufficient constraints for AR(2) stationarity:
    tf = (abs(r2) < 1) && ((r1 + r2) < 1) && ((r2 - r1) < 1);
end

%% FFBS for AR(2) with external noise scaling
function Nhat_new = sample_ar2_states_ffbs_ext(y_target, rho1, rho2, sigma_eps2, ...
    pi_t, alpha, pi_tm1, E_pi_tp1, kappa, x_t, theta, sigma_v2, r_target_scale)

    T = length(y_target);
    if T < 3, Nhat_new = y_target; return; end

    F = [rho1, rho2; 1, 0];
    Q = [sigma_eps2, 0; 0, 0];
    m = zeros(2,T); P = zeros(2,2,T);
    m_pred = zeros(2,T); P_pred = zeros(2,2,T);

    m(:,1) = [y_target(1); 0];
    P(:,:,1) = eye(2)*10;

    for t=2:T
        if t>2
            m_pred(:,t) = F*m(:,t-1);
            P_pred(:,:,t) = F*P(:,:,t-1)*F' + Q;
        else
            m_pred(:,t) = m(:,t-1);
            P_pred(:,:,t) = P(:,:,t-1);
        end

        % Observation 1: target proxy Nhat ~ y_target (small noise)
        H1 = [1,0];
        R1 = sigma_eps2 * r_target_scale;

        % Observation 2: NKPC-implied relation for Nhat_t
        nkpc_obs = alpha*pi_tm1(t) + (1-alpha)*E_pi_tp1(t) + kappa*x_t(t) - pi_t(t);
        H2 = [theta, 0];
        R2 = sigma_v2;

        H = [H1; H2];
        y = [y_target(t); nkpc_obs];
        R = diag([R1, R2]);

        S = H * P_pred(:,:,t) * H' + R;
        K = P_pred(:,:,t) * H' / S;
        m(:,t) = m_pred(:,t) + K*(y - H*m_pred(:,t));
        P(:,:,t) = P_pred(:,:,t) - K*H*P_pred(:,:,t);
    end

    % Backward sampling
    Nhat_states = zeros(2,T);
    Nhat_states(:,T) = mvnrnd(m(:,T), force_pd(P(:,:,T)))';
    for t = T-1:-1:1
        if t>=2
            A = P(:,:,t) * F' / P_pred(:,:,t+1);
            m_s = m(:,t) + A*(Nhat_states(:,t+1) - m_pred(:,t+1));
            P_s = P(:,:,t) - A*(P_pred(:,:,t+1) - P(:,:,t))*A';
            Nhat_states(:,t) = mvnrnd(m_s, force_pd(P_s))';
        else
            Nhat_states(:,t) = Nhat_states(:,t+1);
        end
    end
    Nhat_new = Nhat_states(1,:)';
end

%% FFBS for RW trend with external noise scaling
function Nbar_new = sample_rw_states_ffbs_ext(y_target, n_drift, sigma_eta2, r_rw_scale)
    T = length(y_target);
    if T < 2, Nbar_new = y_target; return; end

    m = zeros(T,1); P = zeros(T,1);
    m(1) = y_target(1); P(1) = 10;

    for t=2:T
        m_pred = n_drift + m(t-1);
        P_pred = P(t-1) + sigma_eta2;
        R_obs  = sigma_eta2 * r_rw_scale;
        K = P_pred / (P_pred + R_obs);
        m(t) = m_pred + K*(y_target(t) - m_pred);
        P(t) = (1-K)*P_pred;
    end

    Nbar_new = zeros(T,1);
    Nbar_new(T) = m(T) + sqrt(max(P(T),1e-8))*randn;
    for t = T-1:-1:1
        A = P(t)/(P(t)+sigma_eta2);
        m_s = m(t) + A*(Nbar_new(t+1) - n_drift - m(t));
        P_s = P(t) * (1 - A);
        Nbar_new(t) = m_s + sqrt(max(P_s,1e-8))*randn;
    end
end

function S = force_pd(S)
    [V,D] = eig((S+S')/2);
    D = diag(max(diag(D),1e-10));
    S = V*D*V';
end