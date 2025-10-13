function [results] = estimate_hsa(pi_data, pi_prev_data, Epi_data, x_data, N_data, n_burn, n_keep)
% Estimates NKPC HSA using state-space model with FFBS Gibbs sampler
    %% Step 1: Data preparation
    T = length(pi_data);
    pi_t = pi_data;
    pi_tm1 = pi_prev_data;
    E_pi_tp1 = Epi_data;
    x_t = x_data;
    %% Step 2: Prior specifications
    % NKPC parameters
    mu_alpha = 0.5; sigma_alpha = 0.1;     % α ~ N(0.5,0.1)
    mu_kappa = 0; sigma_kappa = 0.01;      % κ ~ N(0,0.01)
    mu_theta = 0; sigma_theta = 0.01;      % θ ~ N(0,0.01)
    % State equation parameters  
    mu_rho1 = 0.5; sigma_rho1 = 0.1;     % ρ_1 ~ N(0.5, 0.1)
    mu_rho2 = -0.5; sigma_rho2 = 0.1;    % ρ_2 ~ N(0.5, 0.1)
    mu_n = 0; sigma_n = 0.05;            % n ~ N(0, 0.1²)
    % Variance parameters
    a_v = 0.001; b_v = 0.001;            % σ_v² ~ InvGamma(2.1, 1)
    a_eps = 0.001; b_eps = 0.001;        % σ_ε² ~ InvGamma(2.1, 0.5)  
    a_eta = 0.001; b_eta = 0.001;        % σ_η² ~ InvGamma(0.001, 0.1)
    %% Step 3: Initialize parameters and states
    % Parameters
    alpha = 0.6;
    kappa = 0.3;
    theta = 0.5;
    rho1 = 0.5;
    rho2 = -0.5;
    n = 0.01;
    sigma_v2 = 1.0;
    sigma_eps2 = 0.5;
    sigma_eta2 = 0.1;
    % States: initialize using decomposition of observed N_data
    Nbar = zeros(T, 1);  % N̄_t (trend)
    Nhat = zeros(T, 1);  % N̂_t (cycle)
    % Initialize trend as simple smooth version of N_data
    Nbar(1) = N_data(1);
    Nbar(2) = N_data(2);
    for t = 3:T
        Nbar(t) = 0.7*Nbar(t-1) + 0.3*N_data(t); 
    end
    % Initialize cycle as residual
    Nhat = N_data - Nbar;
    %% Step 4: Storage for MCMC draws  
    alpha_draws = zeros(n_keep, 1);
    kappa_draws = zeros(n_keep, 1);
    theta_draws = zeros(n_keep, 1);
    rho1_draws = zeros(n_keep, 1);
    rho2_draws = zeros(n_keep, 1);
    n_draws = zeros(n_keep, 1);
    sigma_v2_draws = zeros(n_keep, 1);
    sigma_eps2_draws = zeros(n_keep, 1);
    sigma_eta2_draws = zeros(n_keep, 1);
    % Store some state draws for diagnostics
    Nbar_draws = zeros(n_keep, T);
    Nhat_draws = zeros(n_keep, T);
    %% Step 5: FFBS Gibbs Sampling Loop
    fprintf('Starting Gibbs sampling: burn-in=%d, keep=%d\n', n_burn, n_keep);
    for iter = 1:(n_burn + n_keep)
        %% Step 5a: Sample NKPC parameters (α, κ, θ) | states, other params
        % NKPC: π_t = α*π_{t-1} + (1-α)*E_t[π_{t+1}] + κ*x_t - θ*N̂_t + v_t
        % Sample α
        y_alpha = pi_t - E_pi_tp1 - kappa*x_t + theta*Nhat;
        X_alpha = pi_tm1 - E_pi_tp1;
        if abs(X_alpha' * X_alpha) > 1e-12
            prior_precision = 1 / (sigma_alpha^2);
            data_precision = (X_alpha' * X_alpha) / sigma_v2;
            post_precision = prior_precision + data_precision;
            post_variance = 1 / post_precision;
            post_mean = post_variance * (prior_precision * mu_alpha + (X_alpha' * y_alpha) / sigma_v2);
            alpha_draw = post_mean + sqrt(post_variance) * randn;
            alpha = alpha_draw;
        end
        % Sample κ
        y_kappa = pi_t - alpha*pi_tm1 - (1-alpha)*E_pi_tp1 + theta*Nhat;
        X_kappa = x_t;
        if abs(X_kappa' * X_kappa) > 1e-12
            prior_prec_k = 1/sigma_kappa^2;
            data_prec_k = (X_kappa'*X_kappa)/sigma_v2;
            post_prec_k = prior_prec_k + data_prec_k;
            post_var_k = 1/post_prec_k;
            post_mean_k = post_var_k * (prior_prec_k*mu_kappa + (X_kappa'*y_kappa)/sigma_v2);
            kappa_draw = post_mean_k + sqrt(post_var_k)*randn;
            kappa = kappa_draw;
        end
        % Sample θ  
        y_theta = pi_t - alpha*pi_tm1 - (1-alpha)*E_pi_tp1 - kappa*x_t;
        X_theta = -Nhat;  % Note negative sign
        if abs(X_theta' * X_theta) > 1e-12
            prior_prec_t = 1/sigma_theta^2;
            data_prec_t = (X_theta'*X_theta)/sigma_v2;
            post_prec_t = prior_prec_t + data_prec_t;
            post_var_t = 1/post_prec_t;
            post_mean_t = post_var_t * (prior_prec_t*mu_theta + (X_theta'*y_theta)/sigma_v2);
            theta_draw = post_mean_t + sqrt(post_var_t)*randn;
            theta = theta_draw;
        end
        %% Step 5b: Sample state equation parameters (ρ_1, ρ_2, n) | states
        % Sample ρ_1, ρ_2 from AR(2) equation: N̂_t = ρ_1*N̂_{t-1} + ρ_2*N̂_{t-2} + ε_t
        y_rho = Nhat(3:end);  % N̂_t for t=3,...,T
        X_rho = [Nhat(2:end-1), Nhat(1:end-2)];  % [N̂_{t-1}, N̂_{t-2}]
        if size(X_rho, 1) > 0 && abs(det(X_rho'*X_rho)) > 1e-12
            % Prior precision matrix for [ρ_1; ρ_2]
            prior_prec_rho = diag([1/sigma_rho1^2, 1/sigma_rho2^2]);
            data_prec_rho = (X_rho'*X_rho)/sigma_eps2;
            post_prec_rho = prior_prec_rho + data_prec_rho;
            post_cov_rho = inv(post_prec_rho);
            prior_mean_rho = [mu_rho1; mu_rho2];
            post_mean_rho = post_cov_rho * (prior_prec_rho*prior_mean_rho + (X_rho'*y_rho)/sigma_eps2);
            rho_draw = mvnrnd(post_mean_rho, post_cov_rho)';
            rho1 = rho_draw(1);
            rho2 = rho_draw(2);
            % Check stationarity constraint
            max_tries = 2000;
            ok = false;
            for tries = 1:max_tries
                rho_draw = mvnrnd(post_mean_rho, post_cov_rho)';
                if (abs(rho2) < 1) && ((rho1 + rho2) < 1) && ((rho2 - rho1) < 1);
                    rho1 = rho_draw(1);
                    rho2 = rho_draw(2);
                    ok = true;
                    break;
                end
            end
            if ~ok
                rho_try = post_mean_rho;
                shrink  = 0.99;
                rho1 = rho_try(1);
                rho2 = rho_try(2);
            end
        end
        % Sample n from random walk: N̄_t = n + N̄_{t-1} + η_t
        y_n = Nbar(2:end) - Nbar(1:end-1);  % ΔN̄_t = N̄_t - N̄_{t-1}
        T_n = length(y_n);
        if T_n > 0
            prior_prec_n = 1/sigma_n^2;
            data_prec_n = T_n/sigma_eta2;
            post_prec_n = prior_prec_n + data_prec_n;
            post_var_n = 1/post_prec_n;
            post_mean_n = post_var_n * (prior_prec_n*mu_n + sum(y_n)/sigma_eta2);
            n = post_mean_n + sqrt(post_var_n)*randn;
        end
        %% Step 5c: Sample variance parameters
        % Sample σ_v²
        nkpc_resid = pi_t - alpha*pi_tm1 - (1-alpha)*E_pi_tp1 - kappa*x_t + theta*Nhat;
        a_post_v = a_v + T/2;
        b_post_v = b_v + 0.5*sum(nkpc_resid.^2);
        sigma_v2 = 1/gamrnd(a_post_v, 1/b_post_v);
        % Sample σ_ε²
        if length(Nhat) >= 3
            ar_resid = Nhat(3:end) - rho1*Nhat(2:end-1) - rho2*Nhat(1:end-2);
            a_post_eps = a_eps + length(ar_resid)/2;
            b_post_eps = b_eps + 0.5*sum(ar_resid.^2);
            sigma_eps2 = 1/gamrnd(a_post_eps, 1/b_post_eps);
        end
        % Sample σ_η²
        if length(Nbar) >= 2
            rw_resid = Nbar(2:end) - n - Nbar(1:end-1);
            a_post_eta = a_eta + length(rw_resid)/2;
            b_post_eta = b_eta + 0.5*sum(rw_resid.^2);
            sigma_eta2 = 1/gamrnd(a_post_eta, 1/b_post_eta);
        end
    
        %% Step 5d: Sample states using FFBS
        % Sample N̂_t (cycle component) using FFBS for AR(2) process
        Nhat = sample_ar2_states_ffbs(N_data - Nbar, rho1, rho2, sigma_eps2, ...
                                      pi_t, alpha, pi_tm1, E_pi_tp1, kappa, x_t, theta, sigma_v2);
        % Sample N̄_t (trend component) using FFBS for random walk
        Nbar = sample_rw_states_ffbs(N_data - Nhat, n, sigma_eta2);
        %% Store draws (after burn-in)
        if iter > n_burn
            idx = iter - n_burn;
            alpha_draws(idx) = alpha;
            kappa_draws(idx) = kappa;
            theta_draws(idx) = theta;
            rho1_draws(idx) = rho1;
            rho2_draws(idx) = rho2;
            n_draws(idx) = n;
            sigma_v2_draws(idx) = sigma_v2;
            sigma_eps2_draws(idx) = sigma_eps2;
            sigma_eta2_draws(idx) = sigma_eta2;
            % Store every 10th draw to save memory
            if mod(idx, 10) == 1
                store_idx = ceil(idx/10);
                if store_idx <= size(Nbar_draws, 1)
                    Nbar_draws(store_idx, :) = Nbar';
                    Nhat_draws(store_idx, :) = Nhat';
                end
            end
        end
        % Progress report
        if mod(iter, 500) == 0
            fprintf('Iter %d/%d: α=%.3f, κ=%.3f, θ=%.3f, ρ₁=%.3f, ρ₂=%.3f\n', ...
                iter, n_burn+n_keep, alpha, kappa, theta, rho1, rho2);
        end
    end
    %% Step 6: Compute results
    fprintf('\nComputing posterior statistics...\n');
    results = struct();
    % Parameter results
    param_names = {'alpha', 'kappa', 'theta', 'rho1', 'rho2', 'n', 'sigma_v2', 'sigma_eps2', 'sigma_eta2'};
    param_draws = {alpha_draws, kappa_draws, theta_draws, rho1_draws, rho2_draws, n_draws, ...
                   sigma_v2_draws, sigma_eps2_draws, sigma_eta2_draws};
    for i = 1:length(param_names)
        results.(param_names{i}).draws = param_draws{i};
        results.(param_names{i}).mean = mean(param_draws{i});
        results.(param_names{i}).std = std(param_draws{i});
        results.(param_names{i}).quantiles = quantile(param_draws{i}, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]);
    end
    % State results
    results.states.Nbar_mean = mean(Nbar_draws(1:min(end, n_keep/10), :), 1)';
    results.states.Nhat_mean = mean(Nhat_draws(1:min(end, n_keep/10), :), 1)';
    results.states.N_mean = results.states.Nbar_mean + results.states.Nhat_mean;
    % Model diagnostics
    results.diagnostics.T = T;
    results.diagnostics.n_params = 9;
    %% Step 7: Display results
    fprintf('\n=== NKPC Estimation Results ===\n');
    fprintf('Parameter  Mean     Std      2.5%%     97.5%%\n');
    fprintf('alpha      %.4f   %.4f   %.4f   %.4f\n', ...
        results.alpha.mean, results.alpha.std, ...
        results.alpha.quantiles(1), results.alpha.quantiles(7));
    fprintf('kappa      %.4f   %.4f   %.4f   %.4f\n', ...
        results.kappa.mean, results.kappa.std, ...
        results.kappa.quantiles(1), results.kappa.quantiles(7));
    fprintf('sigma_v^2  %.4f   %.4f   %.4f   %.4f\n', ...
        results.sigma_v2.mean, results.sigma_v2.std, ...
        results.sigma_v2.quantiles(1), results.sigma_v2.quantiles(7));
    %% Step 8: Compute Chib (1995) marginal likelihood
    fprintf('\n=== Computing Chib (1995) Marginal Likelihood ===\n');
    results.marginal_likelihood = compute_chib_ces_ml(pi_data, pi_prev_data, Epi_data, x_data, results, 1000, 1000);    
    fprintf('Log Marginal Likelihood: %.4f\n', results.marginal_likelihood.log_ml);
end


%% Sample AR(2) states using Forward-Filtering Backward-Sampling
function Nhat_new = sample_ar2_states_ffbs(y_target, rho1, rho2, sigma_eps2, ...
                                           pi_t, alpha, pi_tm1, E_pi_tp1, kappa, x_t, theta, sigma_v2) 
    T = length(y_target);
    if T < 3
        Nhat_new = y_target;
        return;
    end
    %% Forward filtering
    % State: [N̂_t; N̂_{t-1}]
    % Transition: [N̂_t; N̂_{t-1}] = [ρ₁ ρ₂; 1 0] * [N̂_{t-1}; N̂_{t-2}] + [ε_t; 0]
    F = [rho1, rho2; 1, 0];     % Transition matrix
    Q = [sigma_eps2, 0; 0, 0];  % State noise covariance
    % Observation equation weights (from NKPC)
    H = [theta; 0];  % Only N̂_t affects inflation (first element of state vector)
    % Initialize
    m = zeros(2, T);     % Filtered means
    P = zeros(2, 2, T);  % Filtered covariances
    m_pred = zeros(2, T);
    P_pred = zeros(2, 2, T);
    % Initial conditions (diffuse)
    m(:, 1) = [y_target(1); 0];
    P(:, :, 1) = eye(2) * 10;
    for t = 2:T
        % Predict
        if t > 2
            m_pred(:, t) = F * m(:, t-1);
            P_pred(:, :, t) = F * P(:, :, t-1) * F' + Q;
        else
            m_pred(:, t) = m(:, t-1);
            P_pred(:, :, t) = P(:, :, t-1);
        end
        % Update using both target observation and NKPC constraint
        % Target observation: N̂_t ≈ y_target(t)
        H_target = [1, 0];
        R_target = sigma_eps2 * 0.1;  % Small noise for target matching
        % NKPC observation: π_t = α*π_{t-1} + (1-α)*E_t[π_{t+1}] + κ*x_t - θ*N̂_t + v_t
        % Rearranged: θ*N̂_t = α*π_{t-1} + (1-α)*E_t[π_{t+1}] + κ*x_t - π_t + v_t
        nkpc_obs = alpha*pi_tm1(t) + (1-alpha)*E_pi_tp1(t) + kappa*x_t(t) - pi_t(t);
        H_nkpc = [theta, 0];
        R_nkpc = sigma_v2;
        % Combined update
        H_comb = [H_target; H_nkpc];
        y_comb = [y_target(t); nkpc_obs];
        R_comb = diag([R_target, R_nkpc]);
        % Kalman update
        S = H_comb * P_pred(:, :, t) * H_comb' + R_comb;
        K = P_pred(:, :, t) * H_comb' / S;
        m(:, t) = m_pred(:, t) + K * (y_comb - H_comb * m_pred(:, t));
        P(:, :, t) = P_pred(:, :, t) - K * H_comb * P_pred(:, :, t);
    end
    %% Backward sampling
    Nhat_states = zeros(2, T);
    Nhat_states(:, T) = mvnrnd(m(:, T), P(:, :, T))';
    for t = (T-1):-1:1
        if t >= 2
            % Backward recursion
            A = P(:, :, t) * F' / P_pred(:, :, t+1);
            m_smooth = m(:, t) + A * (Nhat_states(:, t+1) - m_pred(:, t+1));
            P_smooth = P(:, :, t) - A * (P_pred(:, :, t+1) - P(:, :, t)) * A';
            
            % Ensure positive definiteness
            [V, D] = eig(P_smooth);
            D = diag(max(diag(D), 1e-8));
            P_smooth = V * D * V';
            
            Nhat_states(:, t) = mvnrnd(m_smooth, P_smooth)';
        else
            Nhat_states(:, t) = Nhat_states(:, t+1);  % Simple extrapolation
        end
    end
    Nhat_new = Nhat_states(1, :)';  % Extract N̂_t (first component)
end

%% Sample random walk states using FFBS
function Nbar_new = sample_rw_states_ffbs(y_target, n, sigma_eta2)
    T = length(y_target);
    if T < 2
        Nbar_new = y_target;
        return;
    end
    %% Forward filtering for random walk with drift
    % N̄_t = n + N̄_{t-1} + η_t
    m = zeros(T, 1);  % Filtered means
    P = zeros(T, 1);  % Filtered variances
    % Initial condition
    m(1) = y_target(1);
    P(1) = 10;        % Diffuse prior
    for t = 2:T
        % Predict
        m_pred = n + m(t-1);
        P_pred = P(t-1) + sigma_eta2;
        % Update with target observation
        R_obs = sigma_eta2 * 0.1;  % Small observation noise
        % Kalman update  
        K = P_pred / (P_pred + R_obs); % Kalman gain
        m(t) = m_pred + K * (y_target(t) - m_pred);
        P(t) = P_pred * (1 - K);
    end
    %% Backward sampling
    Nbar_new = zeros(T, 1);
    Nbar_new(T) = m(T) + sqrt(P(T)) * randn;
    for t = (T-1):-1:1
        % Backward recursion
        A = P(t) / (P(t) + sigma_eta2);
        m_smooth = m(t) + A * (Nbar_new(t+1) - n - m(t));
        P_smooth = P(t) * (1 - A);
        Nbar_new(t) = m_smooth + sqrt(max(P_smooth, 1e-8)) * randn;
    end
end