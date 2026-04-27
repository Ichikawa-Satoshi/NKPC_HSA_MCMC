function [log_ML, log_components] = func_calc_marginal_likelihood(pi_data, pi_prev_data, Epi_data, x_data, ...
                                        Nhat_data, Nbar_data, priors, results, opts)
% Computes Marginal Likelihood using Chib (1995) / Chib & Jeliazkov (2001) method
% Based on the decomposition of the posterior ordinate (Eq 5 in Chib & Jeliazkov 2001)
%
% log m(y) = log f(y|Theta*) + log pi(Theta*) - log pi(Theta*|y)

    %% 1. Data & Setup
    T = numel(pi_data);
    pi_t     = pi_data(:);
    pi_tm1   = pi_prev_data(:);
    E_pi_tp1 = Epi_data(:);
    x_t      = x_data(:);
    Nhat     = Nhat_data(:);
    Nbar     = Nbar_data(:);
    dNbar_cum = Nbar - Nbar(1);

    % Unpack Priors
    if nargin < 7 || isempty(priors), priors = struct(); end
    mu_alpha    = getd(priors,'mu_alpha',    0.5);
    sigma_alpha = getd(priors,'sigma_alpha', 0.2);
    mu_theta    = getd(priors,'mu_theta',    0.0);
    sigma_theta = getd(priors,'sigma_theta', 0.3);
    mu_kappa0   = getd(priors,'mu_kappa0',   0.3);
    sigma_kappa0= getd(priors,'sigma_kappa0',0.2);
    mu_delta    = getd(priors,'mu_delta',    0.0);
    sigma_delta = getd(priors,'sigma_delta', 0.3);
    a_v         = getd(priors,'a_v',         2.0);
    b_v         = getd(priors,'b_v',         2.0);

    % Unpack Theta* (Posterior Means from results)
    alpha_star  = results.alpha.mean;
    theta_star  = results.theta.mean;
    kappa0_star = results.kappa0.mean;
    delta_star  = results.delta.mean;
    sv2_star    = results.sigma_v2.mean;
    
    % MCMC settings for Reduced Runs
    if nargin < 9 || isempty(opts), opts = struct(); end
    n_reduced = getd(opts, 'n_reduced', 5000); % Sample size for reduced runs
    verbose   = getd(opts, 'verbose', true);

    if verbose, fprintf('Calculating Marginal Likelihood (Chib & Jeliazkov 2001)...\n'); end

    %% 2. Log Likelihood at Theta*
    % Model: pi_t = alpha*pi_{t-1} + (1-alpha)E_t pi_{t+1} + kappa_t x_t - theta Nhat_t + eps_t
    % eps_t = pi_t - alpha*pi_{t-1} - (1-alpha)E_t - kappa_t*x_t + theta*Nhat_t
    
    kappa_t_star = kappa0_star + delta_star * dNbar_cum;
    eps_star = pi_t - alpha_star.*pi_tm1 - (1-alpha_star).*E_pi_tp1 ...
               - kappa_t_star.*x_t + theta_star.*Nhat; % <--- FIXED SIGN
    
    % Normal Likelihood
    log_lik = - (T/2)*log(2*pi) - (T/2)*log(sv2_star) - (1/(2*sv2_star)) * sum(eps_star.^2);

    %% 3. Log Prior at Theta*
    log_prior = 0;
    log_prior = log_prior + log_normpdf(alpha_star, mu_alpha, sigma_alpha);
    log_prior = log_prior + log_normpdf(theta_star, mu_theta, sigma_theta);
    log_prior = log_prior + log_normpdf(kappa0_star, mu_kappa0, sigma_kappa0);
    log_prior = log_prior + log_normpdf(delta_star, mu_delta, sigma_delta);
    log_prior = log_prior + log_invgampdf(sv2_star, a_v, b_v);

    %% 4. Log Posterior Ordinate: log pi(Theta*|y)
    % Decomposition: pi(alpha*) * pi(theta*|alpha*) * pi(kappa*|alpha*,theta*) * pi(sv2*|...)
    
    % --- Term 1: pi(alpha* | y) ---
    % Integrate out theta, kappa, sigma using draws from the MAIN run.
    % We assume we have the draws. If not, we'd need to re-run. 
    % Here we assume 'results' has the full draws.
    
    draws_theta  = results.theta.draws;
    draws_kappa0 = results.kappa0.draws;
    draws_delta  = results.delta.draws;
    draws_sv2    = results.sigma_v2.draws;
    M = numel(draws_theta);
    
    pdf_vals = zeros(M, 1);
    for g = 1:M
        % Reconstruct conditional for alpha given theta^(g), kappa^(g), sv2^(g)
        k_t = draws_kappa0(g) + draws_delta(g) * dNbar_cum;
        y_a = pi_t - E_pi_tp1 - k_t.*x_t + draws_theta(g).*Nhat; % Fixed sign
        X_a = pi_tm1 - E_pi_tp1;
        
        prec0 = 1/(sigma_alpha^2);
        precD = (X_a'*X_a)/draws_sv2(g);
        postV = 1/(prec0 + precD);
        postM = postV*(prec0*mu_alpha + (X_a'*y_a)/draws_sv2(g));
        
        pdf_vals(g) = normpdf(alpha_star, postM, sqrt(postV));
    end
    pi_alpha_star = mean(pdf_vals);
    
    % --- Term 2: pi(theta* | y, alpha*) ---
    % REDUCED RUN 1: Fix alpha = alpha*, sample others.
    if verbose, fprintf('  Reduced Run 1 (Fix alpha)...\n'); end
    
    % Initialize
    th = theta_star; k0 = kappa0_star; d = delta_star; sv2 = sv2_star;
    pdf_vals_2 = zeros(n_reduced, 1);
    
    for i = 1:n_reduced
        % Sample theta (Not strictly needed for the chain, but part of Gibbs)
        % Actually, for Reduced Run 1, we sample theta, kappa, delta, sv2 given alpha*.
        
        % 1. Update theta | alpha*, ...
        % (We calculate the density ordinate at theta* BEFORE updating theta in the chain, 
        %  or average the density of theta* over the draws of the other params in this run.
        %  Standard Chib: Average p(theta* | others) over draws of others.)
        
        k_t = k0 + d * dNbar_cum;
        y_th = pi_t - alpha_star.*pi_tm1 - (1-alpha_star).*E_pi_tp1 - k_t.*x_t; % Fixed sign
        X_th = -Nhat;
        
        prec0 = 1/(sigma_theta^2);
        precD = (X_th'*X_th)/sv2;
        postV = 1/(prec0 + precD);
        postM = postV*(prec0*mu_theta + (X_th'*y_th)/sv2);
        
        % Calculate density at theta* given current k, d, sv2
        pdf_vals_2(i) = normpdf(theta_star, postM, sqrt(postV));
        
        % Draw new theta for the chain
        th = postM + sqrt(postV)*randn;
        
        % 2. Update kappa0, delta | alpha*, theta (current), sv2
        resid = pi_t - alpha_star.*pi_tm1 - (1-alpha_star).*E_pi_tp1 + th.*Nhat;
        X_kappa = [x_t, dNbar_cum.*x_t];
        Sigma0_inv = diag([1/sigma_kappa0^2, 1/sigma_delta^2]);
        mu0 = [mu_kappa0; mu_delta];
        Sigma_post_inv = Sigma0_inv + (X_kappa'*X_kappa)/sv2;
        Sigma_post = inv(Sigma_post_inv);
        mu_post = Sigma_post * (Sigma0_inv*mu0 + (X_kappa'*resid)/sv2);
        params = mu_post + chol(Sigma_post,'lower')*randn(2,1);
        k0 = params(1); d = params(2);
        
        % 3. Update sv2 | alpha*, theta, kappa
        k_t = k0 + d * dNbar_cum;
        eps = pi_t - alpha_star.*pi_tm1 - (1-alpha_star).*E_pi_tp1 - k_t.*x_t + th.*Nhat;
        a_post = a_v + T/2;
        b_post = b_v + 0.5*sum(eps.^2);
        sv2 = 1/gamrnd(a_post, 1/b_post);
    end
    pi_theta_star = mean(pdf_vals_2);
    
    % --- Term 3: pi(kappa0*, delta* | y, alpha*, theta*) ---
    % REDUCED RUN 2: Fix alpha=alpha*, theta=theta*. Sample kappa, delta, sv2.
    if verbose, fprintf('  Reduced Run 2 (Fix alpha, theta)...\n'); end
    
    % Initialize
    k0 = kappa0_star; d = delta_star; sv2 = sv2_star;
    pdf_vals_3 = zeros(n_reduced, 1);
    
    for i = 1:n_reduced
        % 1. Update kappa, delta | alpha*, theta*, sv2
        % (Calculate density first)
        resid = pi_t - alpha_star.*pi_tm1 - (1-alpha_star).*E_pi_tp1 + theta_star.*Nhat;
        X_kappa = [x_t, dNbar_cum.*x_t];
        Sigma0_inv = diag([1/sigma_kappa0^2, 1/sigma_delta^2]);
        mu0 = [mu_kappa0; mu_delta];
        Sigma_post_inv = Sigma0_inv + (X_kappa'*X_kappa)/sv2;
        Sigma_post = inv(Sigma_post_inv);
        mu_post = Sigma_post * (Sigma0_inv*mu0 + (X_kappa'*resid)/sv2);
        
        % Evaluate multivariate normal pdf at [kappa0*; delta*]
        pdf_vals_3(i) = mvnormpdf([kappa0_star; delta_star], mu_post, Sigma_post);
        
        % Draw for chain
        params = mu_post + chol(Sigma_post,'lower')*randn(2,1);
        k0 = params(1); d = params(2);
        
        % 2. Update sv2 | alpha*, theta*, kappa
        k_t = k0 + d * dNbar_cum;
        eps = pi_t - alpha_star.*pi_tm1 - (1-alpha_star).*E_pi_tp1 - k_t.*x_t + theta_star.*Nhat;
        a_post = a_v + T/2;
        b_post = b_v + 0.5*sum(eps.^2);
        sv2 = 1/gamrnd(a_post, 1/b_post);
    end
    pi_kappa_star = mean(pdf_vals_3);

    % --- Term 4: pi(sv2* | y, alpha*, theta*, kappa*) ---
    % No simulation needed. Analytical form available because it's the last block.
    k_t_star = kappa0_star + delta_star * dNbar_cum;
    eps_star = pi_t - alpha_star.*pi_tm1 - (1-alpha_star).*E_pi_tp1 ...
               - k_t_star.*x_t + theta_star.*Nhat;
    
    a_post_star = a_v + T/2;
    b_post_star = b_v + 0.5*sum(eps_star.^2);
    
    pi_sv2_star = exp(log_invgampdf(sv2_star, a_post_star, b_post_star));
    
    %% 5. Final Calculation
    log_posterior = log(pi_alpha_star) + log(pi_theta_star) + ...
                    log(pi_kappa_star) + log(pi_sv2_star);
    
    log_ML = log_lik + log_prior - log_posterior;
    
    if verbose
        fprintf('  LogLik: %.2f, LogPrior: %.2f, LogPost: %.2f\n', log_lik, log_prior, log_posterior);
        fprintf('  Marginal Likelihood: %.2f\n', log_ML);
    end

    % Struct for debugging
    log_components.lik = log_lik;
    log_components.prior = log_prior;
    log_components.post_alpha = log(pi_alpha_star);
    log_components.post_theta = log(pi_theta_star);
    log_components.post_kappa = log(pi_kappa_star);
    log_components.post_sv2   = log(pi_sv2_star);

end

%% ====== Helper Functions ============================================
function val = getd(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f))
        val = s.(f);
    else
        val = d;
    end
end

function l = log_normpdf(x, mu, sigma)
    l = -0.5 * log(2*pi) - log(sigma) - 0.5 * ((x - mu)./sigma).^2;
end

function l = mvnormpdf(x, mu, Sigma)
    k = length(x);
    R = chol(Sigma); % Sigma = R'*R
    logDetSigma = 2 * sum(log(diag(R)));
    diff = x - mu;
    % Mahalanobis distance
    mahal = sum((diff' / R').^2); % equivalent to diff' * inv(Sigma) * diff
    l = -0.5 * (k * log(2*pi) + logDetSigma + mahal);
end

function l = log_invgampdf(x, a, b)
    % f(x) = b^a / gamma(a) * x^(-a-1) * exp(-b/x)
    l = a * log(b) - gammaln(a) - (a + 1) * log(x) - b / x;
end