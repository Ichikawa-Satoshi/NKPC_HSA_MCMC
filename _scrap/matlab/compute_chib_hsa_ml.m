function marginal_likelihood = compute_chib_hsa_ml(pi_data, pi_prev_data, Epi_data, x_data, results, n_burn, n_reduced)
    % COMPUTE_CHIB_STATESPACE_ML - Chib (1995) marginal likelihood for state-space Phillips curve model
 
    % Implements Chib (1995) method with latent states using Carter-Kohn FFBS
    %
    % Inputs:
    %   pi_data      : inflation data (T x 1)
    %   pi_prev_data : lagged inflation (T x 1)
    %   Epi_data     : expected future inflation (T x 1)
    %   x_data       : output gap or other regressor (T x 1)
    %   results      : struct with posterior draws (fields: alpha,kappa,theta,...,Nhat,Nbar)
    %   n_burn       : burn-in samples
    %   n_reduced    : iterations for reduced Gibbs runs
    %
    % Output:
    %   marginal_likelihood : struct with log marginal likelihood and components

    fprintf('=== Chib (1995) Marginal Likelihood Computation ===\n');
    %% Step 1: θ* (posterior means)
    theta_star = struct();
    theta_star.alpha       = results.alpha.mean;
    theta_star.kappa       = results.kappa.mean;
    theta_star.theta       = results.theta.mean;
    theta_star.sigma_v2    = results.sigma_v2.mean;
    theta_star.rho1        = results.rho1.mean;
    theta_star.rho2        = results.rho2.mean;
    theta_star.sigma_eps2  = results.sigma_eps2.mean;
    theta_star.n           = results.n.mean;
    theta_star.sigma_eta2  = results.sigma_eta2.mean;

    %% Step 2: Data
    pi_t    = pi_data(:);
    pi_tm1  = pi_prev_data(:);
    Epi_tp1 = Epi_data(:);
    x_t     = x_data(:);
    T       = length(pi_t);

    %% Step 3: Priors
    pri = struct();
    pri.mu_alpha = 0.5; pri.sigma_alpha = 0.1;
    pri.mu_kappa = 0;   pri.sigma_kappa = 0.01;
    pri.mu_theta = 0;   pri.sigma_theta = 0.01;
    pri.a_v = 0.001;    pri.b_v = 0.001;
    pri.mu_rho1 = 0;    pri.sigma_rho1 = 0.5;
    pri.mu_rho2 = 0;    pri.sigma_rho2 = 0.5;
    pri.a_eps = 0.001;  pri.b_eps = 0.001;
    pri.mu_n = 0;       pri.sigma_n = 1;
    pri.a_eta = 0.001;  pri.b_eta = 0.001;

    %% Step 4: log f(y|θ*) via Kalman filter
    loglik_star = kalman_loglik(pi_t, pi_tm1, Epi_tp1, x_t, theta_star);

    %% Step 5: log prior
    logprior_star = compute_statespace_log_prior(theta_star, pri);
    
    %% Step 6: Posterior ordinate π(θ*|y)
    draws = struct();
    fields = {'alpha','kappa','theta','sigma_v2','rho1','rho2','sigma_eps2','n','sigma_eta2'};
    for f = 1:length(fields)
        draws.(fields{f}) = results.(fields{f}).draws((n_burn+1):end);
    end
    draws.Nhat = results.Nhat.draws((n_burn+1):end,:);
    draws.Nbar = results.Nbar.draws((n_burn+1):end,:);
    G = length(draws.alpha);

    % 6a: π(α*|y)
    log_alpha = zeros(G,1);
    for g=1:G
        y_alpha = pi_t - Epi_tp1 - draws.kappa(g)*x_t + draws.theta(g)*draws.Nhat(g,:)';
        X_alpha = pi_tm1 - Epi_tp1;
        [m_a,v_a] = posterior_norm_params(X_alpha,y_alpha,draws.sigma_v2(g),pri.mu_alpha,pri.sigma_alpha^2);
        log_alpha(g) = log_norm_pdf(theta_star.alpha,m_a,v_a);
    end
    logpost_alpha = stable_logsumexp(log_alpha)-log(G);    
    logpost_kappa = chib_block_kappa(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced);
    logpost_theta = chib_block_theta(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced);
    logpost_sigma_v2 = chib_block_sigma_v2(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced);
    logpost_rho1 = chib_block_rho1(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced);
    logpost_rho2 = chib_block_rho2(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced);
    logpost_sigma_eps2 = chib_block_sigma_eps2(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced);
    logpost_n = chib_block_n(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced);
    logpost_sigma_eta2 = chib_block_sigma_eta2(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced);
    logpost_star = logpost_alpha + logpost_kappa + logpost_theta + logpost_sigma_v2 + logpost_sigma_eps2 + logpost_n + logpost_sigma_eta2;                   
    %% Step 7: Marginal likelihood
    log_ml = loglik_star + logprior_star - logpost_star;
    marginal_likelihood = struct();
    marginal_likelihood.log_ml = log_ml;
    marginal_likelihood.log_likelihood_star = loglik_star;
    marginal_likelihood.log_prior_star = logprior_star;
    marginal_likelihood.log_posterior_star = logpost_star;
    marginal_likelihood.components = struct('alpha',logpost_alpha,'kappa',logpost_kappa,'theta',logpost_theta,...
        'sigma_v2',logpost_sigma_v,'rho1',logpost_rho1,'rho2',logpost_rho2,...
        'sigma_eps2',logpost_sigma_eps,'n',logpost_n,'sigma_eta2',logpost_sigma_eta);
    marginal_likelihood.theta_star = theta_star;
end
%% FFBS (Carter–Kohn)
function [Nhat,Nbar] = ffbs(pi_t,pi_tm1,Epi_tp1,x_t,theta,state)
    T = length(pi_t);
    rho1=theta.rho1; rho2=theta.rho2;
    sig_eps2=state.sigma_eps2; sig_eta2=state.sigma_eta2;
    sig_v2=state.sigma_v2; n=state.n;
    theta_pc=theta.theta; alpha=theta.alpha; kappa=theta.kappa;

    F=[rho1,rho2,0;1,0,0;0,0,1];
    c=[0;0;n];
    Q=diag([sig_eps2,0,sig_eta2]);
    H=[-theta_pc,0,0];
    det_part = alpha*pi_tm1 + (1-alpha)*Epi_tp1 + kappa*x_t;

    s_pred=zeros(3,T); P_pred=zeros(3,3,T);
    s_filt=zeros(3,T); P_filt=zeros(3,3,T);

    s_filt(:,1)=zeros(3,1);
    P_filt(:,:,1)=1e6*eye(3);

    for t=1:T
        if t>1
            s_pred(:,t)=F*s_filt(:,t-1)+c;
            P_pred(:,:,t)=F*P_filt(:,:,t-1)*F'+Q;
        else
            s_pred(:,t)=s_filt(:,1);
            P_pred(:,:,t)=P_filt(:,:,1);
        end
        ytilde=pi_t(t)-det_part(t);
        R=sig_v2;
        S=H*P_pred(:,:,t)*H'+R;
        K=P_pred(:,:,t)*H'/S;
        s_filt(:,t)=s_pred(:,t)+K*(ytilde-H*s_pred(:,t));
        P_filt(:,:,t)=P_pred(:,:,t)-K*H*P_pred(:,:,t);
    end

    s_draw=zeros(3,T);
    s_draw(:,T)=mvnrnd(s_filt(:,T),P_filt(:,:,T))';
    for t=T-1:-1:1
        J=P_filt(:,:,t)*F'/(P_pred(:,:,t+1));
        m=s_filt(:,t)+J*(s_draw(:,t+1)-s_pred(:,t+1));
        V=P_filt(:,:,t)-J*P_pred(:,:,t+1)*J';
        s_draw(:,t)=mvnrnd(m,V)';
    end

    Nhat=s_draw(1,:)';
    Nbar=s_draw(3,:)';
end
%% Kalman likelihood
function loglik = kalman_loglik(pi_t,pi_tm1,Epi_tp1,x_t,theta)
    T=length(pi_t);
    rho1=theta.rho1; rho2=theta.rho2;
    sig_eps2=theta.sigma_eps2; sig_eta2=theta.sigma_eta2;
    sig_v2=theta.sigma_v2; n=theta.n;
    theta_pc=theta.theta; alpha=theta.alpha; kappa=theta.kappa;
    F=[rho1,rho2,0;1,0,0;0,0,1];
    c=[0;0;n];
    Q=diag([sig_eps2,0,sig_eta2]);
    H=[-theta_pc,0,0];
    det_part = alpha*pi_tm1 + (1-alpha)*Epi_tp1 + kappa*x_t;

    s=zeros(3,1); P=1e6*eye(3);
    loglik=0;
    for t=1:T
        s=F*s+c;
        P=F*P*F'+Q;
        ytilde=pi_t(t)-det_part(t);
        S=H*P*H'+sig_v2;
        v=ytilde-H*s;
        loglik=loglik-0.5*(log(2*pi)+log(S)+v^2/S);
        K=P*H'/S;
        s=s+K*v;
        P=P-K*H*P;
    end
end
%% Prior
function logprior = compute_statespace_log_prior(th,pri)
    logprior=0;
    logprior=logprior+log_norm_pdf(th.alpha,pri.mu_alpha,pri.sigma_alpha^2);
    logprior=logprior+log_norm_pdf(th.kappa,pri.mu_kappa,pri.sigma_kappa^2);
    logprior=logprior+log_norm_pdf(th.theta,pri.mu_theta,pri.sigma_theta^2);
    logprior=logprior+log_invgamma_pdf(th.sigma_v2,pri.a_v,pri.b_v);
    logprior=logprior+log_norm_pdf(th.rho1,pri.mu_rho1,pri.sigma_rho1^2);
    logprior=logprior+log_norm_pdf(th.rho2,pri.mu_rho2,pri.sigma_rho2^2);
    logprior=logprior+log_invgamma_pdf(th.sigma_eps2,pri.a_eps,pri.b_eps);
    logprior=logprior+log_norm_pdf(th.n,pri.mu_n,pri.sigma_n^2);
    logprior=logprior+log_invgamma_pdf(th.sigma_eta2,pri.a_eta,pri.b_eta);
end
%% Utils
function lse = stable_logsumexp(x)
    x_max=max(x);
    lse = x_max + log(sum(exp(x - x_max)));
end

%% Block
function logpost_kappa = chib_block_kappa(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced)
% CHIB_BLOCK_KAPPA
%   Computes π(κ* | y, α*) posterior ordinate using
%   "stepwise conditioning (α=α*)" + "reduced Gibbs" + "FFBS".
%
% Inputs:
%   pi_t, pi_tm1, Epi_tp1, x_t : data (T×1)
%   theta_star : struct (alpha,kappa,theta,sigma_v2,rho1,rho2,sigma_eps2,n,sigma_eta2)
%   pri        : prior parameters (mu_*, sigma_*, a_*, b_*)
%   n_reduced  : number of reduced Gibbs iterations J
%
% Output:
%   logpost_kappa : log π(κ* | y, α*)
    T = length(pi_t);

    % Deterministic part of the observation equation under α*
    y_k = pi_t - theta_star.alpha*pi_tm1 - (1 - theta_star.alpha)*Epi_tp1;
    X_k = x_t;

    % Initialize state (parameters and latent states not fixed)
    state.kappa      = theta_star.kappa;      % κ: updated during iterations
    state.theta      = theta_star.theta;      % θ
    state.sigma_v2   = theta_star.sigma_v2;   % σ_v²
    state.rho1       = theta_star.rho1;       % ρ1
    state.rho2       = theta_star.rho2;       % ρ2
    state.sigma_eps2 = theta_star.sigma_eps2; % σ_ε²
    state.n          = theta_star.n;          % n
    state.sigma_eta2 = theta_star.sigma_eta2; % σ_η²
    state.Nhat       = zeros(T,1);            % latent state initial values
    state.Nbar       = zeros(T,1);

    % Store log density values of κ* under conditional distributions
    log_kernel = zeros(n_reduced,1);

    for j = 1:n_reduced
        % ---- (1) σ_v² | α*, κ(cur), θ(cur), states, y  ~ Inv-Gamma
        res_obs = y_k - state.kappa .* X_k + state.theta .* state.Nhat;
        a_v = pri.a_v + T/2;
        b_v = pri.b_v + 0.5 * sum(res_obs.^2);
        state.sigma_v2 = sample_invgamma(a_v, b_v);

        % ---- (2) θ | α*, κ(cur), σ_v²(cur), states, y  ~ Normal
        y_th = y_k - state.kappa .* X_k;
        X_th = -state.Nhat;
        [m_th, v_th] = posterior_norm_params(X_th, y_th, state.sigma_v2, pri.mu_theta, pri.sigma_theta^2);
        state.theta = m_th + sqrt(v_th)*randn;

        % ---- (3) AR(2) part of latent state equation (ρ1, ρ2, σ_ε²)
        if T >= 3
            y_ar = state.Nhat(3:end);
            X1   = state.Nhat(2:end-1);
            X2   = state.Nhat(1:end-2);

            % ρ1 | ρ2, σ_ε²
            XtX = X1'*X1; Xty = X1'*(y_ar - X2*state.rho2);
            v_r1 = 1 / (1/pri.sigma_rho1^2 + XtX/state.sigma_eps2);
            m_r1 = v_r1 * (pri.mu_rho1/pri.sigma_rho1^2 + Xty/state.sigma_eps2);
            state.rho1 = m_r1 + sqrt(v_r1)*randn;

            % ρ2 | ρ1, σ_ε²
            XtX = X2'*X2; Xty = X2'*(y_ar - X1*state.rho1);
            v_r2 = 1 / (1/pri.sigma_rho2^2 + XtX/state.sigma_eps2);
            m_r2 = v_r2 * (pri.mu_rho2/pri.sigma_rho2^2 + Xty/state.sigma_eps2);
            state.rho2 = m_r2 + sqrt(v_r2)*randn;

            % σ_ε² | ρ1,ρ2
            res_eps = y_ar - state.rho1*X1 - state.rho2*X2;
            a_eps = pri.a_eps + (T-2)/2;
            b_eps = pri.b_eps + 0.5*sum(res_eps.^2);
            state.sigma_eps2 = sample_invgamma(a_eps, b_eps);
        end

        % ---- (4) RW part of latent state equation (n, σ_η²)
        if T >= 2
            dN = state.Nbar(2:end) - state.Nbar(1:end-1);
            Xn = ones(T-1,1);
            XtX = Xn'*Xn; Xty = Xn'*dN;
            v_n = 1 / (1/pri.sigma_n^2 + XtX/state.sigma_eta2);
            m_n = v_n * (pri.mu_n/pri.sigma_n^2 + Xty/state.sigma_eta2);
            state.n = m_n + sqrt(v_n)*randn;

            res_eta = dN - state.n;
            a_eta = pri.a_eta + (T-1)/2;
            b_eta = pri.b_eta + 0.5*sum(res_eta.^2);
            state.sigma_eta2 = sample_invgamma(a_eta, b_eta);
        end

        % ---- (5) FFBS (Carter–Kohn): resample latent states
        theta_for_ffbs = struct();
        theta_for_ffbs.rho1  = state.rho1;
        theta_for_ffbs.rho2  = state.rho2;
        theta_for_ffbs.theta = state.theta;
        theta_for_ffbs.alpha = theta_star.alpha;  % fixed at α*
        theta_for_ffbs.kappa = state.kappa;       % current κ
        [state.Nhat, state.Nbar] = ffbs(pi_t, pi_tm1, Epi_tp1, x_t, theta_for_ffbs, state);

        % ---- (6) κ conditional posterior (Normal), evaluate at κ*
        y_kappa = y_k + state.theta .* state.Nhat;
        X_kappa = X_k;
        [m_k, v_k] = posterior_norm_params(X_kappa, y_kappa, state.sigma_v2, pri.mu_kappa, pri.sigma_kappa^2);
        log_kernel(j) = log_norm_pdf(theta_star.kappa, m_k, v_k);

        % Optionally update κ for better mixing
        state.kappa = m_k + sqrt(v_k)*randn;
    end

    % log-mean-exp aggregation
    mlg = max(log_kernel);
    logpost_kappa = mlg + log( mean( exp(log_kernel - mlg) ) );
end

function logpost_theta = chib_block_theta(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced)
% CHIB_BLOCK_THETA
%   Computes π(θ* | y, α*, κ*) posterior ordinate using
%   stepwise conditioning + reduced Gibbs + FFBS.
%
% Inputs:
%   pi_t, pi_tm1, Epi_tp1, x_t : data (T×1)
%   theta_star : struct (alpha,kappa,theta,sigma_v2,rho1,rho2,sigma_eps2,n,sigma_eta2)
%   pri        : prior parameters
%   n_reduced  : number of reduced Gibbs iterations J
%
% Output:
%   logpost_theta : log π(θ* | y, α*, κ*)

    T = length(pi_t);

    % Deterministic part of the observation equation under α*, κ*
    y_th = pi_t - theta_star.alpha*pi_tm1 - (1 - theta_star.alpha)*Epi_tp1 - theta_star.kappa*x_t;
    X_th = -ones(T,1); % placeholder, replaced each iteration with -Nhat

    % Initialize state (parameters not fixed)
    state.theta      = theta_star.theta;      % θ (to be updated)
    state.sigma_v2   = theta_star.sigma_v2;   % σ_v²
    state.rho1       = theta_star.rho1;
    state.rho2       = theta_star.rho2;
    state.sigma_eps2 = theta_star.sigma_eps2;
    state.n          = theta_star.n;
    state.sigma_eta2 = theta_star.sigma_eta2;
    state.Nhat       = zeros(T,1);
    state.Nbar       = zeros(T,1);

    % Store log density values of θ* under conditional distributions
    log_kernel = zeros(n_reduced,1);

    for j = 1:n_reduced
        % ---- (1) σ_v² | α*, κ*, θ(cur), states, y ~ Inv-Gamma
        res_obs = y_th + state.theta .* (-state.Nhat); % y - κx - θNhat
        a_v = pri.a_v + T/2;
        b_v = pri.b_v + 0.5 * sum(res_obs.^2);
        state.sigma_v2 = sample_invgamma(a_v, b_v);

        % ---- (2) AR(2) latent dynamics (ρ1, ρ2, σ_ε²)
        if T >= 3
            y_ar = state.Nhat(3:end);
            X1   = state.Nhat(2:end-1);
            X2   = state.Nhat(1:end-2);

            % ρ1 | ρ2
            XtX = X1'*X1; Xty = X1'*(y_ar - X2*state.rho2);
            v_r1 = 1 / (1/pri.sigma_rho1^2 + XtX/state.sigma_eps2);
            m_r1 = v_r1 * (pri.mu_rho1/pri.sigma_rho1^2 + Xty/state.sigma_eps2);
            state.rho1 = m_r1 + sqrt(v_r1)*randn;

            % ρ2 | ρ1
            XtX = X2'*X2; Xty = X2'*(y_ar - X1*state.rho1);
            v_r2 = 1 / (1/pri.sigma_rho2^2 + XtX/state.sigma_eps2);
            m_r2 = v_r2 * (pri.mu_rho2/pri.sigma_rho2^2 + Xty/state.sigma_eps2);
            state.rho2 = m_r2 + sqrt(v_r2)*randn;

            % σ_ε²
            res_eps = y_ar - state.rho1*X1 - state.rho2*X2;
            a_eps = pri.a_eps + (T-2)/2;
            b_eps = pri.b_eps + 0.5*sum(res_eps.^2);
            state.sigma_eps2 = sample_invgamma(a_eps, b_eps);
        end

        % ---- (3) RW part for Nbar (n, σ_η²)
        if T >= 2
            dN = state.Nbar(2:end) - state.Nbar(1:end-1);
            Xn = ones(T-1,1);
            XtX = Xn'*Xn; Xty = Xn'*dN;
            v_n = 1 / (1/pri.sigma_n^2 + XtX/state.sigma_eta2);
            m_n = v_n * (pri.mu_n/pri.sigma_n^2 + Xty/state.sigma_eta2);
            state.n = m_n + sqrt(v_n)*randn;

            res_eta = dN - state.n;
            a_eta = pri.a_eta + (T-1)/2;
            b_eta = pri.b_eta + 0.5*sum(res_eta.^2);
            state.sigma_eta2 = sample_invgamma(a_eta, b_eta);
        end

        % ---- (4) FFBS (Carter–Kohn): resample latent states
        theta_for_ffbs = struct();
        theta_for_ffbs.rho1  = state.rho1;
        theta_for_ffbs.rho2  = state.rho2;
        theta_for_ffbs.theta = state.theta;       % current θ
        theta_for_ffbs.alpha = theta_star.alpha;  % fixed
        theta_for_ffbs.kappa = theta_star.kappa;  % fixed
        [state.Nhat, state.Nbar] = ffbs(pi_t, pi_tm1, Epi_tp1, x_t, theta_for_ffbs, state);

        % ---- (5) θ conditional posterior (Normal), evaluate at θ*
        y_theta = y_th; 
        X_theta = -state.Nhat;
        [m_th, v_th] = posterior_norm_params(X_theta, y_theta, state.sigma_v2, pri.mu_theta, pri.sigma_theta^2);
        log_kernel(j) = log_norm_pdf(theta_star.theta, m_th, v_th);

        % Optionally update θ for mixing
        state.theta = m_th + sqrt(v_th)*randn;
    end

    % log-mean-exp aggregation
    mlg = max(log_kernel);
    logpost_theta = mlg + log( mean( exp(log_kernel - mlg) ) );
end
function logpost_sigma_v2 = chib_block_sigma_v2(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced)
% CHIB_BLOCK_SIGMA_V2
%   Computes π(σ_v^2* | y, α*, κ*, θ*) posterior ordinate using
%   stepwise conditioning (fix α, κ, θ at their stars) + reduced Gibbs + FFBS.
%
% Inputs:
%   pi_t, pi_tm1, Epi_tp1, x_t : data (T×1)
%   theta_star : struct (alpha,kappa,theta,sigma_v2,rho1,rho2,sigma_eps2,n,sigma_eta2)
%   pri        : prior parameters (mu_*, sigma_*, a_*, b_*)
%   n_reduced  : number of reduced Gibbs iterations J
%
% Output:
%   logpost_sigma_v2 : log π(σ_v^2* | y, α*, κ*, θ*)

    T = length(pi_t);

    % Deterministic part under α*, κ*
    det_part = theta_star.alpha*pi_tm1 + (1 - theta_star.alpha)*Epi_tp1 + theta_star.kappa*x_t;

    % Initialize "remaining" parameters and latent states (current values)
    state.theta      = theta_star.theta;      % θ is fixed in conditioning but we keep a current for FFBS H
    state.sigma_v2   = theta_star.sigma_v2;   % used as measurement variance in FFBS
    state.rho1       = theta_star.rho1;
    state.rho2       = theta_star.rho2;
    state.sigma_eps2 = theta_star.sigma_eps2;
    state.n          = theta_star.n;
    state.sigma_eta2 = theta_star.sigma_eta2;
    state.kappa      = theta_star.kappa;      % fixed; used only for det_part clarity
    state.Nhat       = zeros(T,1);
    state.Nbar       = zeros(T,1);

    % We will average the IG density f_IG(σ_v2* ; a_j, b_j) over j=1..J
    log_kernel = zeros(n_reduced,1);

    for j = 1:n_reduced
        % ---- (1) Update AR(2) part for Nhat: (ρ1, ρ2, σ_ε²) | states
        if T >= 3
            y_ar = state.Nhat(3:end);
            X1   = state.Nhat(2:end-1);
            X2   = state.Nhat(1:end-2);

            % ρ1 | ρ2, σ_ε²  ~ Normal
            XtX = X1'*X1; Xty = X1'*(y_ar - X2*state.rho2);
            v_r1 = 1 / (1/pri.sigma_rho1^2 + XtX/state.sigma_eps2);
            m_r1 = v_r1 * (pri.mu_rho1/pri.sigma_rho1^2 + Xty/state.sigma_eps2);
            state.rho1 = m_r1 + sqrt(v_r1)*randn;

            % ρ2 | ρ1, σ_ε²  ~ Normal
            XtX = X2'*X2; Xty = X2'*(y_ar - X1*state.rho1);
            v_r2 = 1 / (1/pri.sigma_rho2^2 + XtX/state.sigma_eps2);
            m_r2 = v_r2 * (pri.mu_rho2/pri.sigma_rho2^2 + Xty/state.sigma_eps2);
            state.rho2 = m_r2 + sqrt(v_r2)*randn;

            % σ_ε² | ρ1, ρ2  ~ Inv-Gamma
            res_eps = y_ar - state.rho1*X1 - state.rho2*X2;
            a_eps = pri.a_eps + (T-2)/2;
            b_eps = pri.b_eps + 0.5*sum(res_eps.^2);
            state.sigma_eps2 = sample_invgamma(a_eps, b_eps);
        end

        % ---- (2) Update RW part for Nbar: (n, σ_η²) | ΔNbar
        if T >= 2
            dN = state.Nbar(2:end) - state.Nbar(1:end-1);
            Xn = ones(T-1,1);
            XtX = Xn'*Xn; Xty = Xn'*dN;
            v_n = 1 / (1/pri.sigma_n^2 + XtX/state.sigma_eta2);
            m_n = v_n * (pri.mu_n/pri.sigma_n^2 + Xty/state.sigma_eta2);
            state.n = m_n + sqrt(v_n)*randn;

            res_eta = dN - state.n;
            a_eta = pri.a_eta + (T-1)/2;
            b_eta = pri.b_eta + 0.5*sum(res_eta.^2);
            state.sigma_eta2 = sample_invgamma(a_eta, b_eta);
        end

        % ---- (3) FFBS (Carter–Kohn): resample latent states given α*, κ*, θ*
        theta_for_ffbs = struct();
        theta_for_ffbs.rho1  = state.rho1;
        theta_for_ffbs.rho2  = state.rho2;
        theta_for_ffbs.theta = theta_star.theta;     % fixed θ*
        theta_for_ffbs.alpha = theta_star.alpha;     % fixed α*
        theta_for_ffbs.kappa = theta_star.kappa;     % fixed κ*
        [state.Nhat, state.Nbar] = ffbs(pi_t, pi_tm1, Epi_tp1, x_t, theta_for_ffbs, state);

        % ---- (4) Conditional for σ_v² | y, α*, κ*, θ*, states  is Inv-Gamma(a,b)
        % residuals of the observation equation with current latent Nhat:
        %   π_t - [α*π_{t-1} + (1-α*)Eπ_{t+1} + κ*x_t - θ* Nhat_t]
        obs_res = pi_t - det_part + theta_star.theta .* state.Nhat;
        a_v = pri.a_v + T/2;
        b_v = pri.b_v + 0.5 * sum(obs_res.^2);

        % ---- (5) Evaluate IG pdf at σ_v2* and store log-density
        % IG(a,b) pdf: f(x) = b^a / Γ(a) * x^{-(a+1)} * exp(-b/x),  x>0
        xstar = theta_star.sigma_v2;
        log_kernel(j) = a_v*log(b_v) - gammaln(a_v) - (a_v+1)*log(xstar) - b_v/xstar;

        % (Optional) Update σ_v² for the next FFBS iteration's measurement noise
        state.sigma_v2 = sample_invgamma(a_v, b_v);
    end

    % log-mean-exp aggregation over reduced draws
    mlg = max(log_kernel);
    logpost_sigma_v2 = mlg + log( mean( exp(log_kernel - mlg) ) );
end
function logpost_rho1 = chib_block_rho1(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced)
% CHIB_BLOCK_RHO1
%   Computes log π(ρ1* | y, α*, κ*, θ*, σ_v2*) via reduced Gibbs + FFBS.
%   Condition on upstream fixed params (α*, κ*, θ*, σ_v2*).
%
% Output:
%   logpost_rho1 : log posterior ordinate at ρ1*

    T = length(pi_t);

    % Deterministic part under α*, κ*, θ* for FFBS measurement
    det_part = theta_star.alpha*pi_tm1 + (1-theta_star.alpha)*Epi_tp1 + theta_star.kappa*x_t;

    % Initialize current (free) params and latent states
    state.rho1       = theta_star.rho1;       % ρ1 updated
    state.rho2       = theta_star.rho2;       % ρ2 updated
    state.sigma_eps2 = theta_star.sigma_eps2; % σ_ε² updated
    state.n          = theta_star.n;          % n updated
    state.sigma_eta2 = theta_star.sigma_eta2; % σ_η² updated
    state.sigma_v2   = theta_star.sigma_v2;   % fixed for measurement variance
    state.theta      = theta_star.theta;      % fixed in measurement loading
    state.kappa      = theta_star.kappa;      % fixed (in det_part)
    state.Nhat       = zeros(T,1);
    state.Nbar       = zeros(T,1);

    log_kernel = zeros(n_reduced,1);

    for j = 1:n_reduced
        % ---- (1) Update RW part for Nbar: (n, σ_η²)
        if T >= 2
            dN = state.Nbar(2:end) - state.Nbar(1:end-1);
            Xn = ones(T-1,1);
            XtX = Xn'*Xn; Xty = Xn'*dN;
            v_n = 1 / (1/pri.sigma_n^2 + XtX/state.sigma_eta2);
            m_n = v_n * (pri.mu_n/pri.sigma_n^2 + Xty/state.sigma_eta2);
            state.n = m_n + sqrt(v_n)*randn;

            res_eta = dN - state.n;
            a_eta = pri.a_eta + (T-1)/2;
            b_eta = pri.b_eta + 0.5*sum(res_eta.^2);
            state.sigma_eta2 = sample_invgamma(a_eta, b_eta);
        end

        % ---- (2) FFBS under fixed (α*, κ*, θ*) with current state variances
        theta_for_ffbs = struct();
        theta_for_ffbs.rho1  = state.rho1;
        theta_for_ffbs.rho2  = state.rho2;
        theta_for_ffbs.theta = theta_star.theta;     % fixed
        theta_for_ffbs.alpha = theta_star.alpha;     % fixed
        theta_for_ffbs.kappa = theta_star.kappa;     % fixed
        [state.Nhat, state.Nbar] = ffbs(pi_t, pi_tm1, Epi_tp1, x_t, theta_for_ffbs, state);

        % ---- (3) Update AR(2) params (ρ2 | ρ1, σ_ε²) then (σ_ε² | ρ1,ρ2)
        if T >= 3
            y_ar = state.Nhat(3:end);
            X1   = state.Nhat(2:end-1);
            X2   = state.Nhat(1:end-2);

            % ρ2 | ρ1, σ_ε²
            XtX = X2'*X2; Xty = X2'*(y_ar - X1*state.rho1);
            v_r2 = 1 / (1/pri.sigma_rho2^2 + XtX/state.sigma_eps2);
            m_r2 = v_r2 * (pri.mu_rho2/pri.sigma_rho2^2 + Xty/state.sigma_eps2);
            state.rho2 = m_r2 + sqrt(v_r2)*randn;

            % σ_ε² | ρ1, ρ2
            res_eps = y_ar - state.rho1*X1 - state.rho2*X2;
            a_eps = pri.a_eps + (T-2)/2;
            b_eps = pri.b_eps + 0.5*sum(res_eps.^2);
            state.sigma_eps2 = sample_invgamma(a_eps, b_eps);

            % ---- (4) Conditional of ρ1 | ρ2, σ_ε² : Normal; evaluate at ρ1*
            XtX = X1'*X1; Xty = X1'*(y_ar - X2*state.rho2);
            v_r1 = 1 / (1/pri.sigma_rho1^2 + XtX/state.sigma_eps2);
            m_r1 = v_r1 * (pri.mu_rho1/pri.sigma_rho1^2 + Xty/state.sigma_eps2);
            log_kernel(j) = log_norm_pdf(theta_star.rho1, m_r1, v_r1);

            % (Optional) update ρ1 for mixing
            state.rho1 = m_r1 + sqrt(v_r1)*randn;
        else
            % Not enough T to identify AR(2); return prior ordinate
            log_kernel(j) = log_norm_pdf(theta_star.rho1, pri.mu_rho1, pri.sigma_rho1^2);
        end
    end

    mlg = max(log_kernel);
    logpost_rho1 = mlg + log( mean( exp(log_kernel - mlg) ) );
end

function logpost_rho2 = chib_block_rho2(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced)
% CHIB_BLOCK_RHO2
%   Computes log π(ρ2* | y, α*, κ*, θ*, σ_v2*, ρ1*) via reduced Gibbs + FFBS.
%   Now both α*, κ*, θ*, σ_v2*, ρ1* are fixed (upstream).
%
% Output:
%   logpost_rho2 : log posterior ordinate at ρ2*

    T = length(pi_t);

    % Deterministic part under α*, κ*, θ*
    det_part = theta_star.alpha*pi_tm1 + (1-theta_star.alpha)*Epi_tp1 + theta_star.kappa*x_t;

    % Initialize current (free) params and states
    state.rho1       = theta_star.rho1;       % fixed at ρ1*
    state.rho2       = theta_star.rho2;       % ρ2 updated/evaluated
    state.sigma_eps2 = theta_star.sigma_eps2; % σ_ε² updated
    state.n          = theta_star.n;          % n updated
    state.sigma_eta2 = theta_star.sigma_eta2; % σ_η² updated
    state.sigma_v2   = theta_star.sigma_v2;   % fixed for measurement variance
    state.theta      = theta_star.theta;      % fixed
    state.kappa      = theta_star.kappa;      % fixed
    state.Nhat       = zeros(T,1);
    state.Nbar       = zeros(T,1);

    log_kernel = zeros(n_reduced,1);

    for j = 1:n_reduced
        % ---- (1) Update RW part for Nbar
        if T >= 2
            dN = state.Nbar(2:end) - state.Nbar(1:end-1);
            Xn = ones(T-1,1);
            XtX = Xn'*Xn; Xty = Xn'*dN;
            v_n = 1 / (1/pri.sigma_n^2 + XtX/state.sigma_eta2);
            m_n = v_n * (pri.mu_n/pri.sigma_n^2 + Xty/state.sigma_eta2);
            state.n = m_n + sqrt(v_n)*randn;

            res_eta = dN - state.n;
            a_eta = pri.a_eta + (T-1)/2;
            b_eta = pri.b_eta + 0.5*sum(res_eta.^2);
            state.sigma_eta2 = sample_invgamma(a_eta, b_eta);
        end

        % ---- (2) FFBS with fixed (α*, κ*, θ*, ρ1*)
        theta_for_ffbs = struct();
        theta_for_ffbs.rho1  = theta_star.rho1;    % fixed
        theta_for_ffbs.rho2  = state.rho2;         % current
        theta_for_ffbs.theta = theta_star.theta;   % fixed
        theta_for_ffbs.alpha = theta_star.alpha;   % fixed
        theta_for_ffbs.kappa = theta_star.kappa;   % fixed
        [state.Nhat, state.Nbar] = ffbs(pi_t, pi_tm1, Epi_tp1, x_t, theta_for_ffbs, state);

        % ---- (3) Update σ_ε² | ρ1*, ρ2
        if T >= 3
            y_ar = state.Nhat(3:end);
            X1   = state.Nhat(2:end-1);
            X2   = state.Nhat(1:end-2);

            res_eps = y_ar - theta_star.rho1*X1 - state.rho2*X2;
            a_eps = pri.a_eps + (T-2)/2;
            b_eps = pri.b_eps + 0.5*sum(res_eps.^2);
            state.sigma_eps2 = sample_invgamma(a_eps, b_eps);

            % ---- (4) Conditional ρ2 | ρ1*, σ_ε² : Normal; evaluate at ρ2*
            XtX = X2'*X2; Xty = X2'*(y_ar - X1*theta_star.rho1);
            v_r2 = 1 / (1/pri.sigma_rho2^2 + XtX/state.sigma_eps2);
            m_r2 = v_r2 * (pri.mu_rho2/pri.sigma_rho2^2 + Xty/state.sigma_eps2);
            log_kernel(j) = log_norm_pdf(theta_star.rho2, m_r2, v_r2);

            % (Optional) update ρ2 for mixing
            state.rho2 = m_r2 + sqrt(v_r2)*randn;
        else
            log_kernel(j) = log_norm_pdf(theta_star.rho2, pri.mu_rho2, pri.sigma_rho2^2);
        end
    end

    mlg = max(log_kernel);
    logpost_rho2 = mlg + log( mean( exp(log_kernel - mlg) ) );
end

function logpost_sigma_eps2 = chib_block_sigma_eps2(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced)
% CHIB_BLOCK_SIGMA_EPS2
%   Computes log π(σ_ε²* | y, α*, κ*, θ*, σ_v2*, ρ1*, ρ2*) via reduced Gibbs + FFBS.
%   IG(a,b) density evaluated at σ_ε²* with (a,b) from AR(2) residuals each iteration.
%
% Output:
%   logpost_sigma_eps2 : log posterior ordinate at σ_ε²*

    T = length(pi_t);

    % Deterministic part under α*, κ*, θ*
    det_part = theta_star.alpha*pi_tm1 + (1-theta_star.alpha)*Epi_tp1 + theta_star.kappa*x_t;

    % Initialize remaining free params and states
    state.rho1       = theta_star.rho1;       % fixed upstream
    state.rho2       = theta_star.rho2;       % fixed upstream
    state.sigma_eps2 = theta_star.sigma_eps2; % updated then evaluated
    state.n          = theta_star.n;          % updated
    state.sigma_eta2 = theta_star.sigma_eta2; % updated
    state.sigma_v2   = theta_star.sigma_v2;   % fixed for measurement variance
    state.theta      = theta_star.theta;      % fixed
    state.kappa      = theta_star.kappa;      % fixed
    state.Nhat       = zeros(T,1);
    state.Nbar       = zeros(T,1);

    log_kernel = zeros(n_reduced,1);

    for j = 1:n_reduced
        % ---- (1) Update RW part for Nbar
        if T >= 2
            dN = state.Nbar(2:end) - state.Nbar(1:end-1);
            Xn = ones(T-1,1);
            XtX = Xn'*Xn; Xty = Xn'*dN;
            v_n = 1 / (1/pri.sigma_n^2 + XtX/state.sigma_eta2);
            m_n = v_n * (pri.mu_n/pri.sigma_n^2 + Xty/state.sigma_eta2);
            state.n = m_n + sqrt(v_n)*randn;

            res_eta = dN - state.n;
            a_eta = pri.a_eta + (T-1)/2;
            b_eta = pri.b_eta + 0.5*sum(res_eta.^2);
            state.sigma_eta2 = sample_invgamma(a_eta, b_eta);
        end

        % ---- (2) FFBS with fixed (α*, κ*, θ*, ρ1*, ρ2*)
        theta_for_ffbs = struct();
        theta_for_ffbs.rho1  = theta_star.rho1;    % fixed
        theta_for_ffbs.rho2  = theta_star.rho2;    % fixed
        theta_for_ffbs.theta = theta_star.theta;   % fixed
        theta_for_ffbs.alpha = theta_star.alpha;   % fixed
        theta_for_ffbs.kappa = theta_star.kappa;   % fixed
        [state.Nhat, state.Nbar] = ffbs(pi_t, pi_tm1, Epi_tp1, x_t, theta_for_ffbs, state);

        % ---- (3) IG(a,b) for σ_ε² from AR(2) residuals using Nhat
        if T >= 3
            y_ar = state.Nhat(3:end);
            X1   = state.Nhat(2:end-1);
            X2   = state.Nhat(1:end-2);

            res_eps = y_ar - theta_star.rho1*X1 - theta_star.rho2*X2;
            a_eps = pri.a_eps + (T-2)/2;
            b_eps = pri.b_eps + 0.5*sum(res_eps.^2);

            % Evaluate IG pdf at σ_ε²* and store log-density
            xstar = theta_star.sigma_eps2;
            log_kernel(j) = a_eps*log(b_eps) - gammaln(a_eps) - (a_eps+1)*log(xstar) - b_eps/xstar;

            % (Optional) update σ_ε² for next FFBS
            state.sigma_eps2 = sample_invgamma(a_eps, b_eps);
        else
            % Fallback to prior density if T insufficient
            xstar = theta_star.sigma_eps2;
            a0 = pri.a_eps; b0 = pri.b_eps;
            log_kernel(j) = a0*log(b0) - gammaln(a0) - (a0+1)*log(xstar) - b0/xstar;
        end
    end

    mlg = max(log_kernel);
    logpost_sigma_eps2 = mlg + log( mean( exp(log_kernel - mlg) ) );
end

function logpost_n = chib_block_n(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced)
% CHIB_BLOCK_N
%   Computes log π(n* | y, α*, κ*, θ*, σ_v2*, ρ1*, ρ2*, σ_ε²*) via reduced Gibbs + FFBS.
%   Conditional for n in the RW for Nbar is Normal; evaluate at n*.
%
% Output:
%   logpost_n : log posterior ordinate at n*

    T = length(pi_t);

    % Initialize fixed (upstream) and current params
    state.rho1       = theta_star.rho1;      % fixed
    state.rho2       = theta_star.rho2;      % fixed
    state.sigma_eps2 = theta_star.sigma_eps2;% fixed
    state.n          = theta_star.n;         % updated/evaluated
    state.sigma_eta2 = theta_star.sigma_eta2;% updated
    state.sigma_v2   = theta_star.sigma_v2;  % fixed (measurement)
    state.theta      = theta_star.theta;     % fixed
    state.kappa      = theta_star.kappa;     % fixed
    state.Nhat       = zeros(T,1);
    state.Nbar       = zeros(T,1);

    log_kernel = zeros(n_reduced,1);

    for j = 1:n_reduced
        % ---- (1) FFBS with fixed upstream params
        theta_for_ffbs = struct();
        theta_for_ffbs.rho1  = theta_star.rho1;   % fixed
        theta_for_ffbs.rho2  = theta_star.rho2;   % fixed
        theta_for_ffbs.theta = theta_star.theta;  % fixed
        theta_for_ffbs.alpha = theta_star.alpha;  % fixed
        theta_for_ffbs.kappa = theta_star.kappa;  % fixed
        [state.Nhat, state.Nbar] = ffbs(pi_t, pi_tm1, Epi_tp1, x_t, theta_for_ffbs, state);

        if T >= 2
            % ---- (2) n | σ_η², ΔNbar  ~ Normal; evaluate at n*
            dN  = state.Nbar(2:end) - state.Nbar(1:end-1);
            Xn  = ones(T-1,1);
            XtX = Xn'*Xn; Xty = Xn'*dN;
            v_n = 1 / (1/pri.sigma_n^2 + XtX/state.sigma_eta2);
            m_n = v_n * (pri.mu_n/pri.sigma_n^2 + Xty/state.sigma_eta2);
            log_kernel(j) = log_norm_pdf(theta_star.n, m_n, v_n);

            % (Optional) update n and σ_η²
            state.n = m_n + sqrt(v_n)*randn;

            res_eta = dN - state.n;
            a_eta = pri.a_eta + (T-1)/2;
            b_eta = pri.b_eta + 0.5*sum(res_eta.^2);
            state.sigma_eta2 = sample_invgamma(a_eta, b_eta);
        else
            log_kernel(j) = log_norm_pdf(theta_star.n, pri.mu_n, pri.sigma_n^2);
        end
    end

    mlg = max(log_kernel);
    logpost_n = mlg + log( mean( exp(log_kernel - mlg) ) );
end
function logpost_sigma_eta2 = chib_block_sigma_eta2(pi_t, pi_tm1, Epi_tp1, x_t, theta_star, pri, n_reduced)
% CHIB_BLOCK_SIGMA_ETA2
%   Computes log π(σ_η²* | y, α*, κ*, θ*, σ_v2*, ρ1*, ρ2*, σ_ε²*, n*) via reduced Gibbs + FFBS.
%   IG(a,b) density evaluated at σ_η²* where (a,b) from RW residuals of Nbar.
%
% Output:
%   logpost_sigma_eta2 : log posterior ordinate at σ_η²*

    T = length(pi_t);

    % Initialize fixed (upstream) and current params
    state.rho1       = theta_star.rho1;      % fixed
    state.rho2       = theta_star.rho2;      % fixed
    state.sigma_eps2 = theta_star.sigma_eps2;% fixed
    state.n          = theta_star.n;         % fixed
    state.sigma_eta2 = theta_star.sigma_eta2;% updated/evaluated
    state.sigma_v2   = theta_star.sigma_v2;  % fixed (measurement)
    state.theta      = theta_star.theta;     % fixed
    state.kappa      = theta_star.kappa;     % fixed
    state.Nhat       = zeros(T,1);
    state.Nbar       = zeros(T,1);

    log_kernel = zeros(n_reduced,1);

    for j = 1:n_reduced
        % ---- (1) FFBS with all upstream fixed (including n*)
        theta_for_ffbs = struct();
        theta_for_ffbs.rho1  = theta_star.rho1;
        theta_for_ffbs.rho2  = theta_star.rho2;
        theta_for_ffbs.theta = theta_star.theta;
        theta_for_ffbs.alpha = theta_star.alpha;
        theta_for_ffbs.kappa = theta_star.kappa;
        [state.Nhat, state.Nbar] = ffbs(pi_t, pi_tm1, Epi_tp1, x_t, theta_for_ffbs, state);

        % ---- (2) IG(a,b) for σ_η² from RW residuals of Nbar (with n*)
        if T >= 2
            res_eta = (state.Nbar(2:end) - state.Nbar(1:end-1)) - theta_star.n;
            a_eta = pri.a_eta + (T-1)/2;
            b_eta = pri.b_eta + 0.5*sum(res_eta.^2);

            % Evaluate IG at σ_η²*
            xstar = theta_star.sigma_eta2;
            log_kernel(j) = a_eta*log(b_eta) - gammaln(a_eta) - (a_eta+1)*log(xstar) - b_eta/xstar;

            % (Optional) update σ_η² for next FFBS
            state.sigma_eta2 = sample_invgamma(a_eta, b_eta);
        else
            xstar = theta_star.sigma_eta2;
            a0 = pri.a_eta; b0 = pri.b_eta;
            log_kernel(j) = a0*log(b0) - gammaln(a0) - (a0+1)*log(xstar) - b0/xstar;
        end
    end

    mlg = max(log_kernel);
    logpost_sigma_eta2 = mlg + log( mean( exp(log_kernel - mlg) ) );
end