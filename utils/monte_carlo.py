# --------------------------------------------------
# Imports
# --------------------------------------------------

import numpy as np
import emcee

# --------------------------------------------------
# Schechter Function
# --------------------------------------------------

def log_schechter_perdex(logx, logx_star, logphi_star_perdex, alpha):
    """
    Schechter function

    :param logx: Dependent parameter [log(X)]
    :param logx_star: Dependent parameter at knee [log(X)]
    :param logphi_star_perdex: Normalization at knee [Mpc-3 dex-1]
    :param alpha: Power law index of low parameter regime
    :return: Space density of parameter [Mpc-3 dex-1]
    """
    phi_star = 10**logphi_star_perdex
    factor = 10**(logx-logx_star)
    schechter = phi_star*np.exp(-factor)*(factor**(alpha+1))
    return schechter

# --------------------------------------------------
# Monte Carlo Fitting
# --------------------------------------------------

def mcmc(model, logx_vals, y_vals, yerr_vals, fixed_params, fixed_values, logx_min=None, logx_max=None, nwalkers=50, niters=10000, sigma=1e-3, progress=True):
    """
    Fitting Schechter function to data with MCMC
    
    :param model: Choice of "luminosity" or "dust mass"
    :param logx_vals: Dependent parameter values [log(X)]
    :param y_vals: Space density values [Mpc-3 dex-1]
    :param yerr_vals: Error on space density values
    :param fixed_params: List of fixed parameter names (optional)
    :param fixed_values: List of fixed parameter values (optional)
    :param logx_min: Minimum dependent parameter value [log(X)] (optional)
    :param logx_max: Maximum dependent parameter value [log(X)] (optional)
    :param nwalkers: Number of walkers in MC (Default 50)
    :param niters: Number of MC iterations (Default 10000)
    :param sigma: Exploration parameter of MC (Default 0.001)
    :param progress: Show progress (Default True)
    :return: Variable parameter dictionary and MCMC sampler
    """

    # Apply minimum and maximum ranges of fitting if provided
    if logx_min is None:
        logx_min = 0
    if logx_max is None:
        logx_max = 1000

    # Clean input arrays of non-finite values that would cause problems fitting
    logx_vals_cleaned = np.array([x for x,y,yerr in zip(logx_vals, y_vals, yerr_vals) if (np.isfinite(x) and np.isfinite(y) and np.isfinite(yerr) and x!=0 and y!=0 and yerr!=0 and (x>logx_min) and (x<logx_max))])
    y_vals_cleaned = np.array([y for x,y,yerr in zip(logx_vals, y_vals, yerr_vals) if (np.isfinite(x) and np.isfinite(y) and np.isfinite(yerr) and x!=0 and y!=0 and yerr!=0 and (x>logx_min) and (x<logx_max))])
    yerr_vals_cleaned = np.array([yerr for x,y,yerr in zip(logx_vals, y_vals, yerr_vals) if (np.isfinite(x) and np.isfinite(y) and np.isfinite(yerr) and x!=0 and y!=0 and yerr!=0 and (x>logx_min) and (x<logx_max))])

    # Error handling
    if len(fixed_params) != len(fixed_values):
        raise ValueError('Requires an equal number of fixed parameter names to values.')

    # Define the lower and upper bounds of each parameter plus the initial value of the MCMC chain
    logx_star_lower, logx_star_initial, logx_star_upper = 5, 8, 30
    logphi_star_lower, logphi_star_initial, logphi_star_upper = -5, -2.5, 0
    alpha_lower, alpha_initial, alpha_upper = -3., 0., 3.

    # Setting up the arrays for the initial, lower and upper values of parameters
    if model == 'luminosity':
        parameters = ["logl_star", "logphi_star_perdex", "alpha"]
    elif model == 'dust mass':
        parameters = ["logm_star", "logphi_star_perdex", "alpha"]
    parameters_initial = np.array([logx_star_initial, logphi_star_initial, alpha_initial])
    parameters_lower = np.array([logx_star_lower, logphi_star_lower, alpha_lower])
    parameters_upper = np.array([logx_star_upper, logphi_star_upper, alpha_upper])

    # Further error handling
    for it in fixed_params:
        if it not in parameters:
            raise ValueError(str(it) + ' not a valid parameter name. Must be one of: ' + str(parameters))
    if len(fixed_params) >= len(parameters):
        raise ValueError('At least one parameter must be allowed to vary')

    # Identify which parameters the user has defined as fixed and variable
    index_fix = [parameters.index(par) for par in fixed_params]
    vary_params = [param for param in parameters if param not in fixed_params]
    index_vary = [parameters.index(par) for par in vary_params]

    # Initialize the variable parameters
    initial_vary = [parameters_initial[idx] for idx in index_vary]
    lower_vary = [parameters_lower[idx] for idx in index_vary]
    upper_vary = [parameters_upper[idx] for idx in index_vary]
    ndim = len(initial_vary)
    pos = initial_vary + (sigma * np.random.randn(nwalkers, ndim))
    parameters_initial[index_fix] = fixed_values
    data = (parameters_initial, logx_vals_cleaned, y_vals_cleaned, yerr_vals_cleaned)

    def model(theta, theta_full, logx):
        [logx_star, logphi_star, alpha] = theta_full
        y_model = log_schechter_perdex(logx, logx_star, logphi_star, alpha)
        theta_full[index_vary] = theta
        return y_model

    def log_likelihood(theta, theta_full, logx, y, yerr):
        ymodel = model(theta, theta_full, logx)
        log_like = -0.5 * np.sum(((y - ymodel)/yerr)**2)
        return log_like

    def log_prior(theta, theta_full):
        [logx_star, logphi_star, alpha] = theta_full
        theta_full[index_vary] = theta
        for it in range(len(theta)):
            if not lower_vary[it] < theta[it] < upper_vary[it]:
                return -np.inf
        return 0.0

    def log_probability(theta, theta_full, logx, y, yerr):
        lp = log_prior(theta, theta_full)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, theta_full, logx, y, yerr)

    # Run the MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=data, parameter_names=None)
    sampler.run_mcmc(pos, niters, progress=progress)

    return vary_params, sampler

# --------------------------------------------------

def mcmc_array(model, function_results, fixed_params_list=None, fixed_values_list=None, nwalkers=50, niters=10000, sigma=1e-3, logx_min_list=None, logx_max_list=None, progress=True, low_percentile=16, median_percentile=50, high_percentile=84, burn_in=500):
    """
    Fitting Schechter functions to arrays of data with MCMC
    
    :param model: Choice of "luminosity" or "dust mass"
    :param function_results: Dictionary of function results
    :param fixed_params_list: List of fixed parameter names
    :param fixed_values_list: List of fixed parameter values
    :param nwalkers: Number of walkers in MC (Default 50)
    :param niters: Number of MC iterations (Default 10000)
    :param sigma: Exploration parameter of MC (Default 0.001)
    :param logx_min_list: List of minimum parameter values [log(X)]
    :param logx_max_list: List of maximum parameter values [log(X)]
    :param progress: Show progress (Default True)
    :param low_percentile: Low percentile of error on parameters
    :param median_percentile: Median percentile on parameters
    :param high_percentile: High percentile of error on parameters
    :param burn_in: Number of iterations of burn-in
    :return: An array of Schechter parameter values
    """

    vary_params_array = []
    sampler_array = []

    # Error handling (number of fixed parameters must match number of redshift bins)
    if fixed_params_list is not None:
        if len(fixed_params_list) != len(function_results['z_bin_centers']):
            raise ValueError('The names of the fixed parameters does not match the number of redshift slices. If you do not want a fixed parameter for a particular slice add an empty list, [].')
    else:
        fixed_params_list = [[] for _ in range(len(function_results['z_bin_centers']))]

    # Error handling (number of fixed parameters must match number of redshift bins)
    if fixed_values_list is not None:
        if len(fixed_values_list) != len(function_results['z_bin_centers']):
            raise ValueError('The values of the fixed parameters does not match the number of redshift slices. If you do not want a fixed parameter for a particular slice add an empty list, [].')
    else:
        fixed_values_list = [[] for _ in range(len(function_results['z_bin_centers']))]

    # Set no minimum or maximum if not given
    if logx_min_list is None:
        logx_min_list = [None for _ in range(len(function_results['z_bin_centers']))]
    if logx_max_list is None:
        logx_max_list = [None for _ in range(len(function_results['z_bin_centers']))]

    # Monte Carlo fitting for each redshift slice
    logx_vals = function_results['logx_bin_centers']
    for z_it in range(len(function_results['z_bin_centers'])):
        y_vals = function_results['phi_grid'][z_it]
        yerr_vals = function_results['phi_sigma'][z_it]
        fixed_params, fixed_values = fixed_params_list[z_it], fixed_values_list[z_it]
        logx_min, logx_max = logx_min_list[z_it], logx_max_list[z_it]
        vary_params, sampler = mcmc(model, logx_vals, y_vals, yerr_vals, fixed_params, fixed_values, logx_min=logx_min, logx_max=logx_max, nwalkers=nwalkers, niters=niters, sigma=sigma, progress=progress)
        vary_params_array.append(vary_params)
        sampler_array.append(sampler)

    # List of model parameters depending on model
    if model == 'luminosity':
        parameters = ["logl_star", "logphi_star_perdex", "alpha"]
    elif model == 'dust mass':
        parameters = ["logm_star", "logphi_star_perdex", "alpha"]
    else:
        raise ValueError('The first argument, "model", must be one of "luminosity" or "dust mass".')

    percentiles = [low_percentile, median_percentile, high_percentile]
    percentile_name = ['_low', '', '_high']

    # Create a results dictionary
    results = {}
    for i in range(len(parameters)):
        for j in range(len(percentile_name)):
            param = parameters[i]
            id_name = param+percentile_name[j]
            results[id_name] = np.zeros(len(sampler_array))

    # At each redshift sample the parameter space for best values
    for z_it in range(len(sampler_array)):
        sampler = sampler_array[z_it]
        fixed_params = fixed_params_list[z_it]
        fixed_values = fixed_values_list[z_it]
        vary_params = vary_params_array[z_it]

        flat_samples = sampler.get_chain(discard=burn_in, flat=True)
        n = len(flat_samples)
        sample_theta = np.zeros((n, len(parameters)))
        vary_param_idx = [parameters.index(vary_params[param_idx]) for param_idx in range(len(vary_params))]
        fix_param_idx = [parameters.index(fixed_params[param_idx]) for param_idx in range(len(fixed_params))]
        sample_theta[:, vary_param_idx] = flat_samples
        sample_theta[:, fix_param_idx] = fixed_values

        for i in range(len(parameters)):
            for j in range(len(percentile_name)):
                param = parameters[i]
                id_name = param+percentile_name[j]
                mcmc_results = np.percentile(sample_theta[:, i], percentiles[j])
                results[id_name][z_it] = mcmc_results

    # Add sampler, fixed parameters and best fitting parameters to dictionary
    results['sampler_array'] = sampler_array
    results['fixed_params'] = fixed_params_list
    results['fixed_values'] = fixed_values_list
    results['vary_params'] = vary_params_array
    results['logphi_star_low'] = np.log10((10**results['logphi_star_perdex_low'])/np.log(10))
    results['logphi_star'] = np.log10((10**results['logphi_star_perdex'])/np.log(10))
    results['logphi_star_high'] = np.log10((10**results['logphi_star_perdex_high'])/np.log(10))
    return results
