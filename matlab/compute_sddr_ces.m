function sddr = compute_sddr_ces(results)    
    % κ
    mu_kappa = 0; sigma_kappa = 0.01;
    % Posterior density
    [post_kappa_density, x_kappa_post] = ksdensity(results.kappa.draws);
    prior_mean_kappa = mu_kappa;
    post_mean_kappa  = mean(results.kappa.draws);
    % Savage-Dickey ratio for H0: κ = 0
    test_value_kappa = 0;
    prior_at_test = normpdf(test_value_kappa, mu_kappa, sigma_kappa);
    post_at_test = interp1(x_kappa_post, post_kappa_density, test_value_kappa, 'linear', 0);    
    sddr = post_at_test/prior_at_test;
end


