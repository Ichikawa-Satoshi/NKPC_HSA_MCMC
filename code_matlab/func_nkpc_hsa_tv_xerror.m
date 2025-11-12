
function results = func_nkpc_hsa_tv_xerror(pi_data, pi_prev_data, Epi_data, x_data, N_data, n_burn, n_keep, priors, opts)
% Estimates NKPC HSA with time-varying kappa_t via FFBS Gibbs sampler
    %% ---------------- Data ----------------
    T = numel(pi_data);
    pi_t     = pi_data(:);
    pi_tm1   = pi_prev_data(:);
    E_pi_tp1 = Epi_data(:);
    x_t      = x_data(:);
    N_obs    = N_data(:);
    if any([numel(pi_tm1), numel(E_pi_tp1), numel(x_t), numel(N_obs)] ~= T)
        error('All input series must have the same length T.');
    end
    %% ---------------- Priors ----------------
    if nargin < 8 || isempty(priors), priors = struct(); end
    mu_alpha    = getd(priors,'mu_alpha',    0.5);
    sigma_alpha = getd(priors,'sigma_alpha', 0.2);
    mu_theta    = getd(priors,'mu_theta',    0.0);
    sigma_theta = getd(priors,'sigma_theta', 0.3);
    mu_rho1     = getd(priors,'mu_rho1',     0.5);
    sigma_rho1  = getd(priors,'sigma_rho1',  0.2);
    mu_rho2     = getd(priors,'mu_rho2',    -0.5);
    sigma_rho2  = getd(priors,'sigma_rho2',  0.2);
    mu_n        = getd(priors,'mu_n',        0.0);
    sigma_n     = getd(priors,'sigma_n',     0.1);

    % time-varying kappa part
    mu_beta     = getd(priors,'mu_beta',     0.0);
    sigma_beta  = getd(priors,'sigma_beta',  0.3);

    % variances
    a_v         = getd(priors,'a_v',         2.0);
    b_v         = getd(priors,'b_v',         2.0);
    a_eps       = getd(priors,'a_eps',       2.0);
    b_eps       = getd(priors,'b_eps',       2.0);
    a_eta       = getd(priors,'a_eta',       2.0);
    b_eta       = getd(priors,'b_eta',       2.0);
    a_u         = getd(priors,'a_u',         2.0);
    b_u         = getd(priors,'b_u',         2.0);

    assertallpos([sigma_alpha,sigma_theta,sigma_rho1,sigma_rho2,sigma_n, ...
                  a_v,b_v,a_eps,b_eps,a_eta,b_eta,a_u,b_u], ...
                  'Prior stds and IG params must be positive.');

    %% ---------------- Options ----------------
    if nargin < 9 || isempty(opts), opts = struct(); end
    alpha     = getd(opts,'alpha0',     0.6);
    theta     = getd(opts,'theta0',     0.5);
    rho1      = getd(opts,'rho10',      0.5);
    rho2      = getd(opts,'rho20',     -0.5);
    n_drift   = getd(opts,'n0',         0.01);
    beta      = getd(opts,'beta0',      0.0);
    sigma_v2  = getd(opts,'sigma_v20',  1.0);
    sigma_eps2= getd(opts,'sigma_eps20',0.5);
    sigma_eta2= getd(opts,'sigma_eta20',0.1);
    sigma_u2  = getd(opts,'sigma_u20',  0.1);
    seed      = getd(opts,'seed',       []);
    verbose   = getd(opts,'verbose',    true);
    store_every = max(1, getd(opts,'store_every', 1));
    enforce_station   = getd(opts,'enforce_stationary', true);
    r_target_scale    = getd(opts,'r_target_scale', 0.1);
    r_rw_scale        = getd(opts,'r_rw_scale',     0.1);

    if ~isempty(seed), rng(seed); end

    %% ---------------- Initial states ----------------
    % simple trend init
    Nbar = zeros(T,1);
    Nbar(1:min(2,T)) = N_obs(1:min(2,T));
    for t=3:T
        Nbar(t) = 0.7*Nbar(t-1) + 0.3*N_obs(t);
    end
    Nhat = N_obs - Nbar;
    kappa_t = 0.3 * ones(T,1);  % initial time-varying kappa

    %% ---------------- Storage ----------------
    n_store = ceil(n_keep/store_every);
    alpha_draws = zeros(n_store,1);
    theta_draws = zeros(n_store,1);
    beta_draws  = zeros(n_store,1);
    rho1_draws  = zeros(n_store,1);
    rho2_draws  = zeros(n_store,1);
    n_draws     = zeros(n_store,1);
    sv_draws    = zeros(n_store,1);
    se_draws    = zeros(n_store,1);
    su_draws    = zeros(n_store,1);
    seta_draws  = zeros(n_store,1);

    kappa_draws = zeros(n_store,T);
    Nbar_draws  = zeros(n_store,T);
    Nhat_draws  = zeros(n_store,T);

    if verbose
        fprintf('Gibbs start: burn-in=%d, keep=%d\n', n_burn, n_keep);
    end

    %% ---------------- Gibbs loop ----------------
    total_iter = n_burn + n_keep;
    store_idx = 0;
    for iter = 1:total_iter

        % ===== 1. alpha | . =====
        y_a = pi_t - E_pi_tp1 - kappa_t.*x_t + theta.*Nhat;
        X_a = pi_tm1 - E_pi_tp1;
        prec0 = 1/(sigma_alpha^2);
        precD = (X_a'*X_a)/sigma_v2;
        postV = 1/(prec0 + precD);
        postM = postV*(prec0*mu_alpha + (X_a'*y_a)/sigma_v2);
        alpha = postM + sqrt(postV)*randn;

        % ===== 2. theta | . =====
        y_thet = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 - kappa_t.*x_t;
        X_thet = -Nhat;
        prec0 = 1/(sigma_theta^2);
        precD = (X_thet'*X_thet)/sigma_v2;
        postV = 1/(prec0 + precD);
        postM = postV*(prec0*mu_theta + (X_thet'*y_thet)/sigma_v2);
        theta = postM + sqrt(postV)*randn;

        % ===== 3. beta, sigma_u2 | kappa_t, Nbar =====
        if T >= 2
            dNbar  = diff(Nbar);
            dKappa = diff(kappa_t);
            prec0 = 1/(sigma_beta^2);
            precD = sum(dNbar.^2) / sigma_u2;
            postV = 1/(prec0 + precD);
            postM = postV*(prec0*mu_beta + sum(dNbar.*dKappa)/sigma_u2);
            beta  = postM + sqrt(postV)*randn;

            res_u = dKappa - beta*dNbar;
            a_post = a_u + numel(res_u)/2;
            b_post = b_u + 0.5*sum(res_u.^2);
            sigma_u2 = 1/gamrnd(a_post, 1/b_post);
        end

        % ===== 4. AR(2) params for Nhat =====
        if T >= 3
            y_r = Nhat(3:end);
            X_r = [Nhat(2:end-1), Nhat(1:end-2)];
            Prec0 = diag([1/sigma_rho1^2, 1/sigma_rho2^2]);
            PrecD = (X_r'*X_r)/sigma_eps2;
            PostP = Prec0 + PrecD; PostC = inv(PostP);
            mu0 = [mu_rho1; mu_rho2];
            PostM = PostC*(Prec0*mu0 + (X_r'*y_r)/sigma_eps2);
            ok=false; tries=0;
            while ~ok && tries<1000
                rdraw = mvnrnd(PostM, PostC)';
                if ~enforce_station || is_stationary_ar2(rdraw(1), rdraw(2))
                    rho1 = rdraw(1); rho2 = rdraw(2); ok=true;
                end
                tries = tries+1;
            end
        end

        % ===== 5. RW drift n =====
        if T >= 2
            y_n = diff(Nbar);
            prec0 = 1/sigma_n^2;
            precD = numel(y_n)/sigma_eta2;
            postV = 1/(prec0 + precD);
            postM = postV*(prec0*mu_n + sum(y_n)/sigma_eta2);
            n_drift = abs(postM + sqrt(postV)*randn);
        end

        % ===== 6. variance updates =====
        nkpc_resid = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 - kappa_t.*x_t + theta.*Nhat;
        a_post = a_v + T/2;
        b_post = b_v + 0.5*sum(nkpc_resid.^2);
        sigma_v2 = 1/gamrnd(a_post, 1/b_post);

        if T >= 3
            ar_res = Nhat(3:end) - rho1.*Nhat(2:end-1) - rho2.*Nhat(1:end-2);
            a_post = a_eps + numel(ar_res)/2;
            b_post = b_eps + 0.5*sum(ar_res.^2);
            sigma_eps2 = 1/gamrnd(a_post, 1/b_post);
        end

        if T >= 2
            rw_res = diff(Nbar) - n_drift;
            a_post = a_eta + numel(rw_res)/2;
            b_post = b_eta + 0.5*sum(rw_res.^2);
            sigma_eta2 = 1/gamrnd(a_post, 1/b_post);
        end

        % ===== 7. FFBS for states =====
        % (1) cycle Nhat
        Nhat = sample_ar2_states_ffbs_ext(N_obs - Nbar, rho1, rho2, sigma_eps2, ...
            pi_t, alpha, pi_tm1, E_pi_tp1, kappa_t, x_t, theta, sigma_v2, r_target_scale);

        % (2) trend Nbar
        Nbar = sample_rw_states_ffbs_ext(N_obs - Nhat, n_drift, sigma_eta2, r_rw_scale);

        % (3) time-varying kappa
        kappa_t = sample_kappa_tvffbs(pi_t, pi_tm1, E_pi_tp1, n_drift, x_t, ...
                                      Nhat, alpha, theta, ...
                                      beta, sigma_v2, sigma_u2, sigma_eta2);

        % ===== 8. Store =====
        if iter > n_burn && mod(iter - n_burn, store_every)==0
            store_idx = store_idx + 1;
            alpha_draws(store_idx) = alpha;
            theta_draws(store_idx) = theta;
            beta_draws(store_idx)  = beta;
            rho1_draws(store_idx)  = rho1;
            rho2_draws(store_idx)  = rho2;
            n_draws(store_idx)     = n_drift;
            sv_draws(store_idx)    = sigma_v2;
            se_draws(store_idx)    = sigma_eps2;
            seta_draws(store_idx)  = sigma_eta2;
            su_draws(store_idx)    = sigma_u2;
            kappa_draws(store_idx,:) = kappa_t';
            Nbar_draws(store_idx,:)  = Nbar';
            Nhat_draws(store_idx,:)  = Nhat';
        end

        if verbose && mod(iter, 2000)==0
            fprintf('Iter %d/%d: a=%.3f th=%.3f b=%.3f sigv=%.3f\n', ...
                    iter, total_iter, alpha, theta, beta, sigma_v2);
        end
    end

    %% ---------------- Results ----------------
    results = struct();
    results.alpha      = summarize(alpha_draws);
    results.theta      = summarize(theta_draws);
    results.beta       = summarize(beta_draws);
    results.rho1       = summarize(rho1_draws);
    results.rho2       = summarize(rho2_draws);
    results.n          = summarize(n_draws);
    results.sigma_v2   = summarize(sv_draws);
    results.sigma_eps2 = summarize(se_draws);
    results.sigma_eta2 = summarize(seta_draws);
    results.sigma_u2   = summarize(su_draws);

    results.states.kappa_mean = mean(kappa_draws,1)';
    results.states.Nbar_mean  = mean(Nbar_draws,1)';
    results.states.Nhat_mean  = mean(Nhat_draws,1)';
    results.states.N_mean     = results.states.Nbar_mean + results.states.Nhat_mean;
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

function assertallpos(v, msg)
    if any(~isfinite(v) | v <= 0)
        error(msg);
    end
end

function tf = is_stationary_ar2(r1, r2)
    tf = (abs(r2) < 1) && ((r1 + r2) < 1) && ((r2 - r1) < 1);
end

%% ====== kappa FFBS ==================================================
function kappa_path = sample_kappa_tvffbs(pi_t, pi_tm1, E_pi_tp1, n_drift, x_t, ...
                                          Nhat, alpha, theta, ...
                                          beta, sigma_v2, sigma_u2, sigma_eta2)
    T = length(pi_t);
    % y_k = κ_t x_t + noise
    y_k = pi_t - alpha.*pi_tm1 - (1-alpha).*E_pi_tp1 + theta.*Nhat;
    k_filt = zeros(T,1);
    P_filt = zeros(T,1);
    a_pred = zeros(T,1);
    R_pred = zeros(T,1);
    % diffuse init
    k_filt(1) = 0;
    P_filt(1) = 10;
    a_pred(1) = k_filt(1);
    R_pred(1) = P_filt(1);
    % ----- forward -----
    for t = 2:T
        a_t = k_filt(t-1) + beta * n_drift;
        R_t = P_filt(t-1) + beta^2 * sigma_eta2 + sigma_u2;
        H = x_t(t);
        R = sigma_v2;
        v = y_k(t) - H*a_t;
        S = H^2 * R_t + R;
        K = R_t * H / S;
        k_filt(t) = a_t + K*v;
        P_filt(t) = (1 - K*H) * R_t;
        a_pred(t) = a_t;
        R_pred(t) = R_t;
    end

    % ----- backward sampling -----
    kappa_path = zeros(T,1);
    kappa_path(T) = k_filt(T) + sqrt(max(P_filt(T),1e-8)) * randn;
    for t = T-1:-1:1
        Jt = P_filt(t) / R_pred(t+1);
        m_s = k_filt(t) + Jt * (kappa_path(t+1) - a_pred(t+1));
        P_s = P_filt(t) - Jt^2 * (R_pred(t+1) - P_filt(t));
        P_s = max(P_s, 1e-8);
        kappa_path(t) = m_s + sqrt(P_s) * randn;
    end
end

%% ====== AR(2) FFBS for Nhat =========================================
function Nhat_new = sample_ar2_states_ffbs_ext(y_target, rho1, rho2, sigma_eps2, ...
    pi_t, alpha, pi_tm1, E_pi_tp1, kappa_t, x_t, theta, sigma_v2, r_target_scale)
    T = length(y_target);
    if T < 3
        Nhat_new = y_target;
        return;
    end
    F = [rho1, rho2; 1, 0];
    Q = [sigma_eps2, 0; 0, 0];
    m      = zeros(2,T);
    P      = zeros(2,2,T);
    m_pred = zeros(2,T);
    P_pred = zeros(2,2,T);
    % init
    m(:,1)   = [y_target(1); 0];
    P(:,:,1) = eye(2)*10;
    for t = 2:T
        % prediction
        if t > 2
            m_pred(:,t) = F * m(:,t-1);
            P_pred(:,:,t) = F * P(:,:,t-1) * F' + Q;
        else
            m_pred(:,t) = m(:,t-1);
            P_pred(:,:,t) = P(:,:,t-1);
        end
        % obs1
        H1 = [1, 0];
        R1 = sigma_eps2 * r_target_scale;
        % obs2 (NKPC)
        nkpc_obs = alpha*pi_tm1(t) ...
                 + (1-alpha)*E_pi_tp1(t) ...
                 + kappa_t(t)*x_t(t) ...
                 - pi_t(t);
        H2 = [theta, 0];
        R2 = sigma_v2;
        H = [H1; H2];
        y = [y_target(t); nkpc_obs];
        R = diag([R1, R2]);
        S = H * P_pred(:,:,t) * H' + R;
        K = P_pred(:,:,t) * H' / S;
        m(:,t) = m_pred(:,t) + K * (y - H * m_pred(:,t));
        P(:,:,t) = P_pred(:,:,t) - K * H * P_pred(:,:,t);
    end
    % backward
    Nhat_states = zeros(2,T);
    Nhat_states(:,T) = mvnrnd(m(:,T), force_pd(P(:,:,T)))';
    C_s_next = P(:,:,T);
    for t = T-1:-1:1
        if t >= 2
            A = P(:,:,t) * F' / P_pred(:,:,t+1);
            m_s = m(:,t) + A * (Nhat_states(:,t+1) - m_pred(:,t+1));
            P_s = P(:,:,t) - A * (P_pred(:,:,t+1) - C_s_next) * A';
            Nhat_states(:,t) = mvnrnd(m_s, force_pd(P_s))';
            C_s_next = P_s;
        else
            Nhat_states(:,t) = Nhat_states(:,t+1);
        end
    end
    Nhat_new = Nhat_states(1,:)';
end

%% ====== RW FFBS for Nbar ============================================
function Nbar_new = sample_rw_states_ffbs_ext(y_target, n_drift, sigma_eta2, r_rw_scale)
    T = length(y_target);
    if T < 2
        Nbar_new = y_target;
        return;
    end

    m = zeros(T,1);
    P = zeros(T,1);
    m_pred_store = zeros(T,1); 
    P_pred_store = zeros(T,1); 
    m(1) = y_target(1);
    P(1) = 10;
    for t = 2:T
        m_pred = n_drift + m(t-1);
        P_pred = P(t-1) + sigma_eta2;
        R_obs  = sigma_eta2 * r_rw_scale;
        K = P_pred / (P_pred + R_obs);
        m(t) = m_pred + K*(y_target(t) - m_pred);
        P(t) = (1-K)*P_pred;
        m_pred_store(t) = m_pred; 
        P_pred_store(t) = P_pred; 
    end

    Nbar_new = zeros(T,1);
    Nbar_new(T) = m(T) + sqrt(max(P(T),1e-8)) * randn;
    C_next = P(T);

    for t = T-1:-1:1
        Jt = P(t) / P_pred_store(t+1);
        m_s = m(t) + Jt * (Nbar_new(t+1) - m_pred_store(t+1));
        P_s = P(t) - Jt^2 * (P_pred_store(t+1) - C_next);
        P_s = max(P_s, 1e-8);
        Nbar_new(t) = m_s + sqrt(P_s) * randn;
        C_next = P_s;
    end
end

%% PD helper
function S = force_pd(S)
    S = (S+S')/2;
    [V,D] = eig(S);
    D = diag(max(diag(D),1e-10));
    S = V*D*V';
end