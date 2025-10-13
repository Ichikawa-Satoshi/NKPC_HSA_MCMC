function [results] = estimate_ces(pi_data, pi_prev_data, Epi_data, x_data, n_burn, n_keep)
    % Estimates NKPC CES parameters using Gibbs sampling
    %% Step 1: Data preparation
    T = length(pi_data);   
    pi_t = pi_data(1:end);           % π_t (dependent variable)
    pi_tm1 = pi_prev_data(1:end);    % π_{t-1} 
    E_pi_tp1 = Epi_data(1:end);      % E_t[π_{t+1}] (use realized values)
    x_t = x_data(1:end);             % x_t (output gap)
    T_obs = length(pi_t);            % Number of observations for regression
 
    %% Step 2: Prior specifications
    % α ~ Normal(0.5, 1)
    mu_alpha = 0.5; 
    sigma_alpha = 0.1;
    % κ ~ Normal(0, 1)
    mu_kappa = 0;     % Prior mean
    sigma_kappa = 0.01;  % Prior standard deviation (uninformative)
    % σ_v² ~ InvGamma(a_sig, b_sig)
    a_sig = 0.001; 
    b_sig = 0.001;

    %% Step 3: Initialize parameters
    alpha = 0.5;     % Starting value for α
    kappa = 0.1;    % Starting value for κ  
    sigma_v2 = 0.1;  % Starting value for σ_v²    
    fprintf('Initial values: α=%.3f, κ=%.3f, σ_v²=%.3f\n', alpha, kappa, sigma_v2);
    
    %% Step 4: Storage for MCMC draws
    alpha_draws = zeros(n_keep, 1);
    kappa_draws = zeros(n_keep, 1);
    sigma_v2_draws = zeros(n_keep, 1);
    fprintf('\nStarting Gibbs sampling...\n');
    fprintf('Burn-in: %d, Keep: %d\n', n_burn, n_keep);
    
    %% Step 5: Gibbs sampling loop
    for iter = 1:(n_burn + n_keep)
        %% Step 5a: Sample α | κ, σ_v², data
        % Model: π_t = α*π_{t-1} + (1-α)*E_t[π_{t+1}] + κ*x_t + v_t
        % Rearrange: π_t - E_t[π_{t+1}] - κ*x_t = α*(π_{t-1} - E_t[π_{t+1}]) + v_t
        y_alpha = pi_t - E_pi_tp1 - kappa * x_t;  % Left-hand side
        X_alpha = pi_tm1 - E_pi_tp1;              % Right-hand side regressor
        
        % Posterior for α is Normal distribution
        % Prior: α ~ Normal(a_alpha, b_alpha)
        prior_precision = 1 / (sigma_alpha^2);
        data_precision = (X_alpha' * X_alpha) / sigma_v2;
        post_precision = prior_precision + data_precision;
        post_variance = 1 / post_precision;
        post_mean = post_variance * (prior_precision * mu_alpha + (X_alpha' * y_alpha) / sigma_v2);
        alpha_draw = post_mean + sqrt(post_variance) * randn;
        alpha = alpha_draw;
        %% Step 5b: Sample κ | α, σ_v², data  
        % Model: π_t - α*π_{t-1} - (1-α)*E_t[π_{t+1}] = κ*x_t + v_t
        y_kappa = pi_t - alpha * pi_tm1 - (1 - alpha) * E_pi_tp1;
        X_kappa = x_t;
        % Posterior for κ is Normal
        % Prior: κ ~ N(mu_kappa, sigma_kappa²)
        prior_precision = 1 / (sigma_kappa^2);
        data_precision = (X_kappa' * X_kappa) / sigma_v2;
        post_precision = prior_precision + data_precision;
        post_variance = 1 / post_precision;
        post_mean = post_variance * (prior_precision * mu_kappa + (X_kappa' * y_kappa) / sigma_v2);
        % draw candidate
        kappa_draw = post_mean + sqrt(post_variance) * randn;
        kappa = kappa_draw;
        %% Step 5c: Sample σ_v² | α, κ, data
        % Calculate residuals
        residuals = pi_t - alpha * pi_tm1 - (1 - alpha) * E_pi_tp1 - kappa * x_t;
        % Posterior for σ_v² is InvGamma  
        % Prior: σ_v² ~ InvGamma(a_sig, b_sig)
        a_post = a_sig + T_obs / 2;
        b_post = b_sig + 0.5 * sum(residuals.^2);
        % Sample from InvGamma(a_post, b_post)
        sigma_v2 = 1 / gamrnd(a_post, 1/b_post);
        %% Store draws (after burn-in)
        if iter > n_burn
            idx = iter - n_burn;
            alpha_draws(idx) = alpha;
            kappa_draws(idx) = kappa;
            sigma_v2_draws(idx) = sigma_v2;
        end
        
        % Progress report
        if mod(iter, 1000) == 0
            fprintf('Iteration %d/%d: α=%.3f, κ=%.3f, σ_v²=%.3f\n', ...
                iter, n_burn + n_keep, alpha, kappa, sigma_v2);
        end
    end
    %% Step 6: Compute posterior statistics
    fprintf('\nComputing posterior statistics...\n');
    results = struct();
    % Alpha results
    results.alpha.draws = alpha_draws;
    results.alpha.mean = mean(alpha_draws);
    results.alpha.std = std(alpha_draws);
    results.alpha.quantiles = quantile(alpha_draws, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]);
    
    % Kappa results  
    results.kappa.draws = kappa_draws;
    results.kappa.mean = mean(kappa_draws);
    results.kappa.std = std(kappa_draws);
    results.kappa.quantiles = quantile(kappa_draws, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]);
    
    % Sigma_v2 results
    results.sigma_v2.draws = sigma_v2_draws;
    results.sigma_v2.mean = mean(sigma_v2_draws);
    results.sigma_v2.std = std(sigma_v2_draws);  
    results.sigma_v2.quantiles = quantile(sigma_v2_draws, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]);
    
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