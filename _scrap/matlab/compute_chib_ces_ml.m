function marginal_likelihood = compute_chib_ces_ml(pi_data, pi_prev_data, Epi_data, x_data, results, n_burn, n_reduced)
    % COMPUTE_CHIB_MARGINAL_LIKELIHOOD_CORRECTED - Corrected implementation of Chib (1995) method
    fprintf('Starting Chib (1995) marginal likelihood computation...\n');
    
    %% Step 1: Choose high-density point θ* (use posterior means)
    theta_star = struct();
    theta_star.alpha = results.alpha.mean;
    theta_star.kappa = results.kappa.mean;  
    theta_star.sigma_v2 = results.sigma_v2.mean;
    fprintf('High-density point θ*: α=%.4f, κ=%.4f, σ_v²=%.4f\n', ...
        theta_star.alpha, theta_star.kappa, theta_star.sigma_v2);
    
    %% Step 2: Prepare data
    pi_t = pi_data;
    pi_tm1 = pi_prev_data;
    Epitp1 = Epi_data;
    x_t = x_data;
    T = length(pi_t);
    
    %% Step 3: Prior specifications
    pri.mu_alpha = 0.5; pri.sigma_alpha = 0.01;
    pri.mu_kappa = 0.01; pri.sigma_kappa = 0.01;
    pri.a_v = 0.001; pri.b_v = 0.001;
    
    %% Step 4: Compute log likelihood at θ*
    loglik_star = compute_ces_loglikelihood( ...
        pi_t, pi_tm1, Epitp1, x_t, theta_star.alpha, theta_star.kappa, theta_star.sigma_v2);
    
    %% Step 5: Compute log prior at θ*
    logprior_star = compute_ces_log_prior(theta_star, pri);
    
    %% Step 6: Estimation of log posterior ordinate π(θ*|y)
    alpha_samples = results.alpha.draws((n_burn+1):end);
    kappa_samples = results.kappa.draws((n_burn+1):end);
    sigma_v2_samples = results.sigma_v2.draws((n_burn+1):end);
    n_samples = length(alpha_samples);    
    % 6a: Estimate π(α*|y) using full Gibbs samples
    fprintf('Estimating π(α*|y)...\n');
    logpost_alpha_vals = zeros(n_samples, 1);
    for g = 1:n_samples
        kappa_g = kappa_samples(g);
        sigma_v2_g = sigma_v2_samples(g);
        % Complete conditional for α
        y_alpha = pi_t - Epitp1 - kappa_g * x_t;
        X_alpha = pi_tm1 - Epitp1;
        [m_a, v_a] = posterior_norm_params(X_alpha, y_alpha, sigma_v2_g, pri.mu_alpha, pri.sigma_alpha^2);
        logpost_alpha_vals(g) = log_norm_pdf(theta_star.alpha, m_a, v_a);
    end
    logpost_alpha = stable_logsumexp(logpost_alpha_vals) - log(n_samples);    
    % 6b: Estimate π(κ* | y, α*) via reduced Gibbs
    logpost_kappa_vals = zeros(n_reduced, 1);    
    y_k = pi_t - theta_star.alpha*pi_tm1 - (1 - theta_star.alpha)*Epitp1; % T×1
    X_k = x_t;                                                            % T×1    
    kappa_curr  = mean(kappa_samples);
    sigma2_curr = mean(sigma_v2_samples);    
    for j = 1:n_reduced
        % --- σ² | α*, κ_curr, y  ~ Inv-Gamma
        res = y_k - kappa_curr * X_k;
        a_v_post = pri.a_v + T/2;
        b_v_post = pri.b_v + 0.5 * sum(res.^2);
        sigma2_curr = sample_invgamma(a_v_post, b_v_post);    
        % --- κ | α*, σ²_curr, y  ~ Normal
        [m_k, v_k] = posterior_norm_params(X_k, y_k, sigma2_curr, pri.mu_kappa, pri.sigma_kappa^2);   
        kappa_curr = normrnd(m_k, sqrt(v_k));
        logpost_kappa_vals(j) = log_norm_pdf(theta_star.kappa, m_k, v_k);
    end    
    logpost_kappa = stable_logsumexp(logpost_kappa_vals) - log(n_reduced);  
    % 6c: Compute π(σ_v²*|y,α*,κ*)     
    res_v = pi_t - theta_star.alpha*pi_tm1 - (1-theta_star.alpha)*Epitp1 - theta_star.kappa*x_t;
    a_v_post = pri.a_v + T/2;
    b_v_post = pri.b_v + 0.5*sum(res_v.^2);
    logpost_sigma_v = log_invgamma_pdf(theta_star.sigma_v2, a_v_post, b_v_post);    
    
    % Total log posterior ordinate
    logpost_star = logpost_alpha + logpost_kappa + logpost_sigma_v;    
    %% Step 7: Compute log marginal likelihood using BMI
    log_marginal_likelihood = loglik_star + logprior_star - logpost_star;
    
    %% Step 8: Numerical standard error (optional)
    % This would require additional computation following Chib (1995) Section 3
    
    fprintf('=== CORRECTED Chib (1995) Results ===\n');
    fprintf('Log likelihood: %.6f\n', loglik_star);
    fprintf('Log prior:      %.6f\n', logprior_star);
    fprintf('Log posterior components:\n');
    fprintf('  π(α*|y):           %.6f\n', logpost_alpha);
    fprintf('  π(κ*|y,α*):        %.6f\n', logpost_kappa);
    fprintf('  π(σ_v²*|y,α*,κ*):  %.6f\n', logpost_sigma_v);
    fprintf('  Total:             %.6f\n', logpost_star);
    fprintf('Log marginal likelihood: %.6f\n', log_marginal_likelihood);
    
    %% Convergence diagnostics
    fprintf('\nConvergence diagnostics:\n');
    fprintf('α component range: [%.6f, %.6f]\n', min(logpost_alpha_vals), max(logpost_alpha_vals));
    fprintf('κ component range: [%.6f, %.6f]\n', min(logpost_kappa_vals), max(logpost_kappa_vals));
    
    %% Store results
    marginal_likelihood = struct();
    marginal_likelihood.log_ml = log_marginal_likelihood;
    marginal_likelihood.log_likelihood_star = loglik_star;
    marginal_likelihood.log_prior_star = logprior_star;
    marginal_likelihood.log_posterior_star = logpost_star;
    marginal_likelihood.components = struct('alpha', logpost_alpha, 'kappa', logpost_kappa, 'sigma_v2', logpost_sigma_v);
    marginal_likelihood.theta_star = theta_star;
end

%% Improved helper functions
function lse = stable_logsumexp(x)
    % More stable log-sum-exp with underflow protection
    if isempty(x)
        lse = -Inf;
        return;
    end
    
    x_max = max(x);
    if isinf(x_max)
        lse = x_max;
        return;
    end
    
    x_shifted = x - x_max;
    exp_sum = sum(exp(x_shifted));
    
    if exp_sum == 0
        lse = -Inf;
    else
        lse = x_max + log(exp_sum);
    end
end

function x = sample_invgamma(a, b)
    % Sample from Inverse Gamma distribution IG(a,b)
    % More robust implementation
    if a <= 0 || b <= 0
        error('Invalid parameters for inverse gamma distribution');
    end
    x = 1 / gamrnd(a, 1/b);
end

%% helper functions
function loglik = compute_ces_loglikelihood(pi_t, pi_tm1, E_pi_tp1, x_t, alpha, kappa, sigma_v2)
    residuals = pi_t - alpha*pi_tm1 - (1-alpha)*E_pi_tp1 - kappa*x_t;
    T = length(residuals);
    loglik = -0.5*T*log(2*pi) - 0.5*T*log(sigma_v2) - 0.5*sum(residuals.^2)/sigma_v2;
end

function logprior = compute_ces_log_prior(th, pri)
    logprior_kappa = log_norm_pdf(th.kappa, pri.mu_kappa, pri.sigma_kappa^2);
    logprior_alpha = log_norm_pdf(th.alpha, pri.mu_alpha, pri.sigma_alpha^2);
    logprior_sv = log_invgamma_pdf(th.sigma_v2, pri.a_v, pri.b_v);
    logprior = logprior_alpha + logprior_kappa + logprior_sv;
end

function [m, v] = posterior_norm_params(X, y, sigma2, mu0, sigma0_sq)
    XtX = X' * X;
    Xty = X' * y;
    tau0 = 1 / sigma0_sq;
    tau = XtX / sigma2;
    v = 1 / (tau0 + tau);
    m = v * (tau0 * mu0 + Xty / sigma2);
end

function lp = log_norm_pdf(x, m, v)
    lp = -0.5*log(2*pi*v) - 0.5*((x - m).^2)/v;
end

function lp = log_invgamma_pdf(x, a, b)
    if x <= 0
        lp = -Inf;
        return;
    end
    lp = a*log(b) - gammaln(a) - (a+1)*log(x) - b/x;
end