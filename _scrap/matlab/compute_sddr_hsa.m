function sddr = compute_sddr_hsa(results)
    % ===== Normal priors (for plotting only) =====
    mu_kappa = 0;    sig_kappa   = 0.01;
    mu_theta = 0;    sig_theta   = 0.01;
    % Helper: Savage–Dickey BF_01 at 0 with Normal prior (posterior/prior)
    sddr_bf01 = @(post_x, post_f, mu, sig) ...
        ( max(1e-12, interp1(post_x, post_f, 0, 'linear', 0)) / max(1e-12, normpdf(0, mu, sig)) );
    % ---- kappa ----
    [post_k, xk_post] = ksdensity(results.kappa.draws);
    bf01_kappa = sddr_bf01(xk_post, post_k, mu_kappa, sig_kappa);  % BF_01 = post/prior
    % ---- theta ----
    [post_t, xt_post] = ksdensity(results.theta.draws);
    bf01_theta = sddr_bf01(xt_post, post_t, mu_theta, sig_theta);
    sddr = struct('kappa', bf01_kappa, 'theta', bf01_theta);
end