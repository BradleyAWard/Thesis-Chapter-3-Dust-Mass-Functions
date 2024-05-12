# --------------------------------------------------
# Imports
# --------------------------------------------------

import utils
import numpy as np
from astropy import units as u
from tqdm import tqdm
from scipy.interpolate import interp1d

# --------------------------------------------------
# Accessible volume calculation
# --------------------------------------------------

def v(model, method, parameters, wave, z_low, z_high, logx_low, logx_high, T, beta, flux_lim, logx_gal=None, kappa=None):
    """
    Calculates the accessible volume in multiple ways
    
    :param model: Choice of "luminosity" or "dust mass"
    :param method: Choice of "Vmax", "PC00" or "PC00+D11"
    :param parameters: Dictionary containing cosmology and survey area
    :param wave: Target wavelength od function [microns]
    :param z_low: Minimum redshift value
    :param z_high: Maximum redshift value
    :param logx_low: Minimum value of measured value [log(X)]
    :param logx_high: Maximum value of measured value [log(X)]
    :param T: Dust temperature [K]
    :param beta: Dust emissivity spectral index
    :param flux_lim: Limiting flux density of survey [mJy]
    :param logx_gal: Measured value for specific galaxy (use with Vmax)
    :param kappa: Dust mass absorption coefficient (optional) [m2 kg-1]
    :return: The accessible volume [Mpc3]
    """
    # Define the cosmology and size of the survey area
    H0, Om, Ok, Olam = parameters['cosmology']
    omega = parameters['area']
    omega = omega.to(u.steradian)

    # Create a subgrid in z - logX space
    z_small = np.linspace(z_low, z_high, 100)
    if method == 'Vmax':
        logx_small = logx_gal
    elif (method == 'PC00') or (method == 'PC00+D11'):
        logx_small = np.linspace(logx_low, logx_high, 100)
        del_logx = logx_high - logx_low
    x_small = 10**logx_small
    X, Z = np.meshgrid(x_small, z_small, sparse=True)

    # Obtain flux across subgrid from either luminosity or dust mass
    if model == 'luminosity':
        flux = utils.luminosity_to_flux(X, wave, Z, T, beta, parameters['cosmology'])
    elif model == 'dust mass':
        flux = utils.dust_mass_to_flux(X, wave, wave, Z, T, beta, kappa, parameters['cosmology'])

    # Calculate the comoving volume element across the subgrid
    V = omega*utils.differential_comoving_v(Z, H0, Om, Ok, Olam)
    V_limit = np.where(flux.value > flux_lim, V, 0)

    # Integrate across the grid according to the selected method
    if method == 'Vmax':
        V_acc = np.trapz(V_limit, z_small, axis=0).item()
    elif (method == 'PC00') or (method == 'PC00+D11'):
        V_acc = np.trapz(np.trapz(V_limit, z_small, axis=0), logx_small, axis=0)/del_logx

    return V_acc.value

# --------------------------------------------------
# Binned function
# --------------------------------------------------

def get_function(model, method, parameters, wave, z_bins, logx_bins, z_values, logx_values, c_values, t_values, beta_values, flux_lim_values, kappa=None, disable=False, n_min=10):
    """
    Calculates the binned function for any galaxy parameter
    
    :param model: Choice of "luminosity" or "dust mass"
    :param method: Choice of "Vmax", "PC00" or "PC00+D11"
    :param parameters: Dictionary containing cosmology and survey area
    :param wave: Target wavelength of function [microns]
    :param z_bins: Redshift bins
    :param logx_bins: Measured parameter bins [log(x)]
    :param z_values: Redshift values for each galaxy
    :param logx_values: Measured parameter values for each galaxy [log(x)]
    :param c_values: Completeness correction factor for each galaxy
    :param t_values: Dust temperature values for each galaxy [K]
    :param beta_values: Dust emissivity spectral index for each galaxy
    :param flux_lim_values: Survey flux limit for each galaxy [mJy]
    :param kappa: Dust mass absorption coefficient [m2 kg-1] (optional)
    :param disable: Disable progress (Default False)
    :param n_min: Minimum galaxies in a bin to be counted (Default 10)
    :return: Binned function with N,C,V,phi grids
    """
    # Set up bins of z and x
    min_z, max_z, n_z = min(z_bins), max(z_bins), len(z_bins)-1
    min_logx, max_logx, n_logx = min(logx_bins), max(logx_bins), len(logx_bins)-1
    z_bin_centers = (z_bins[:-1] + z_bins[1:])/2
    logx_bin_centers = (logx_bins[:-1] + logx_bins[1:])/2
    dlogx = (max_logx - min_logx)/n_logx

    # Double for loops iterate over each bin in z - logX space
    N_list, C_list, V_list, phi_list = [], [], [], []
    for z_it in tqdm(range(len(z_bin_centers)), desc=method+' (Redshift Slice)', position=0, leave=True, disable=disable):
        for logx_it in tqdm(range(len(logx_bin_centers)), desc=method+' (log X Bin)', position=1, leave=True, disable=disable):

            N = len([z for z, logx in zip(z_values, logx_values) if (z_bins[z_it] < z < z_bins[z_it + 1]) & (logx_bins[logx_it] < logx < logx_bins[logx_it + 1])])
            if N >= n_min:

                # Obtain the number, correction factors, temperatures, betas and flux limits for objects in a given bin
                N = len([z for z, logx in zip(z_values, logx_values) if (z_bins[z_it] < z < z_bins[z_it+1]) & (logx_bins[logx_it] < logx < logx_bins[logx_it+1])])
                C = [c for c, z, logx in zip(c_values, z_values, logx_values) if (z_bins[z_it] < z < z_bins[z_it+1]) & (logx_bins[logx_it] < logx < logx_bins[logx_it+1])]
                T = [t for t, z, logx in zip(t_values, z_values, logx_values) if (z_bins[z_it] < z < z_bins[z_it+1]) & (logx_bins[logx_it] < logx < logx_bins[logx_it+1])]
                Beta = [beta for beta, z, logx in zip(beta_values, z_values, logx_values) if (z_bins[z_it] < z < z_bins[z_it+1]) & (logx_bins[logx_it] < logx < logx_bins[logx_it+1])]
                Flux_lim = [flux for flux, z, logx in zip(flux_lim_values, z_values, logx_values) if (z_bins[z_it] < z < z_bins[z_it+1]) & (logx_bins[logx_it] < logx < logx_bins[logx_it+1])]

                # If using the 1/Vmax or PC00+D11 method, calculate phi for each individual galaxy
                V_bin, phi_bin = [], []
                if method == 'Vmax' or method == 'PC00+D11':
                    if method == 'Vmax':
                        Logx = [logx for z, logx in zip(z_values, logx_values) if (z_bins[z_it] < z < z_bins[z_it+1]) & (logx_bins[logx_it] < logx < logx_bins[logx_it+1])]
                    elif method == 'PC00+D11':
                        Logx = [None for z, logx in zip(z_values, logx_values) if (z_bins[z_it] < z < z_bins[z_it+1]) & (logx_bins[logx_it] < logx < logx_bins[logx_it+1])]
                    for gal in range(N):
                        V_gal = v(model, method, parameters, wave, z_bins[z_it], z_bins[z_it+1], logx_bins[logx_it], logx_bins[logx_it+1], T[gal], Beta[gal], Flux_lim[gal], logx_gal=Logx[gal], kappa=kappa)
                        phi_gal = C[gal]/V_gal
                        V_bin.append(V_gal)
                        phi_bin.append(phi_gal)

                    # Note this is not a physical volume, it is a median of all objects for comparison purposes.
                    V_list.append(np.nanmedian(V_bin))
                    phi_list.append(np.nansum(phi_bin))

                # If using the PC00 method, calculate phi with an averaged volume over the bin
                elif method == 'PC00':
                    T_median, Beta_median, Flux_lim_median = np.nanmedian(T), np.nanmedian(Beta), np.nanmedian(Flux_lim)
                    V = v(model, method, parameters, wave, z_bins[z_it], z_bins[z_it+1], logx_bins[logx_it], logx_bins[logx_it+1], T_median, Beta_median, Flux_lim_median, logx_gal=None, kappa=kappa)
                    phi = np.nansum(C)/V
                    V_list.append(V)
                    phi_list.append(phi)
                else:
                    raise ValueError('Method must be one of "Vmax", "PC00" or "PC00+D11".')

                N_list.append(N)
                C_list.append(np.nansum(C))

            else:
                N_list.append(np.nan)
                C_list.append(np.nan)
                V_list.append(np.nan)
                phi_list.append(np.nan)

    # Reshape the lists into an array with pre-specified bin numbers
    N_grid = np.array(N_list).reshape(n_z, n_logx)
    C_grid = np.array(C_list).reshape(n_z, n_logx)
    V_grid = np.array(V_list).reshape(n_z, n_logx)
    phi_grid = np.array(phi_list).reshape(n_z, n_logx)/dlogx

    # Append all important arrays to a dictionary
    results = {'z_bin_centers': z_bin_centers,
                'logx_bin_centers': logx_bin_centers,
                'N_grid': N_grid,
                'C_grid': C_grid,
                'V_grid': V_grid,
                'phi_grid': phi_grid}

    return results

# --------------------------------------------------
# Survey Class
# --------------------------------------------------

class Survey:
    """
    Class for a survey of galaxies
    """
    def __init__(self, galaxies, parameters):

        # Make a check that the following are present in parameters
        if 'cosmology' not in parameters:
            raise ValueError('Please provide "cosmology" in parameters: [H0 (km/s/Mpc), Omega_m, Omega_k, Omega_lambda]')
        if 'area' not in parameters:
            raise ValueError('Please provide "area" in parameters: The area of the survey [square degrees].')
        if 'submm correction' not in parameters:
            print('No corrections for the sub-mm catalogue are supplied. If they should be, please provide "submm correction" in parameters with the tuple: (bins, correction factors).')
        if 'id correction' not in parameters:
            print('No corrections for the ID analysis are supplied. If they should be, please provide "id correction" in parameters with the tuple: (bins, correction factors).')
        self.parameters = parameters
        self.galaxies = galaxies

        self.function_run = False
        self.function_fit = False

    # Get redshifts of all galaxies (including randomized for errors)
    def get_redshifts(self, randomized_z=False):
        """
        Obtains redshifts of galaxies
        
        :param randomized_z: Choice to randomized redshifts (optional)
        :return: All redshifts in survey
        """
        redshifts = np.array([gal.redshift for gal in self.galaxies])
        if randomized_z:
            redshift_errors = np.array([gal.redshift_err for gal in self.galaxies])
            redshifts = np.abs(np.random.normal(redshifts, redshift_errors))
        return redshifts

    # Get redshift errors of all galaxies
    def get_redshift_errors(self):
        """
        Obtains redshift errors of galaxies
        
        :return: All redshift errors in survey
        """
        redshift_errors = np.array([gal.redshift_err for gal in self.galaxies])
        return redshift_errors

    # Get fluxes of all galaxies at a given wavelength
    def get_fluxes(self, wave):
        """
        Obtains flux densities of galaxies
        
        :param wave: Wavelength of flux densities [microns]
        :return: All flux densities in survey
        """
        f = np.array([self.galaxies[gal].flux_jy[self.galaxies[gal].wavelengths_obs_um == wave].item() for gal in range(len(self.galaxies))])
        return f

    # Get flux errors of all galaxies at a given wavelength (Pre calibration error - as used to determine detection)
    def get_flux_errors(self, wave):
        """
        Obtains flux density errors of galaxies
        
        :param wave: Wavelength of flux densities [microns]
        :return: All flux density errors in survey
        """
        f_err = np.array([self.galaxies[gal].flux_error_jy[self.galaxies[gal].wavelengths_obs_um == wave].item() for gal in range(len(self.galaxies))])
        return f_err

    # Get the flux limit of the survey for all galaxies (Can be different for each galaxy if based on SNR)
    def get_flux_limits(self, wave, fixed_limit=None, fixed_snr=None):
        """
        Obtains flux density limits of galaxies
        
        :param wave: Wavelength of flux limit [microns]
        :param fixed_limit: Fixed flux limit for all galaxies (optional)
        :param fixed_snr: Fixed SNR limit for all galaxies (optional)
        :return: All flux density limits in survey
        """
        if fixed_limit is not None:
            f_lim = np.array([fixed_limit for _ in range(len(self.galaxies))])
        elif fixed_snr is not None:
            flux_errors = self.get_flux_errors(wave)
            f_lim = np.array([f_err*fixed_snr for f_err in flux_errors])
        else:
            f_lim = 0.0294
        return f_lim

    # Get dust temperatures of all galaxies
    def get_dust_temperatures(self, fixed_t=None, t_inputs=None):
        """
        Obtains dust temperatures of galaxies
        
        :param fixed_t: Fixed dust temperature for all galaxies (optional)
        :param t_inputs: Set dust temperatures for all galaxies (optional)
        :return: All dust temperatures in survey
        """
        if fixed_t is not None:
            t = np.array([fixed_t for _ in range(len(self.galaxies))])
        elif t_inputs is not None:
            t = np.array(t_inputs)
        else:
            t = np.array([self.galaxies[gal].sed_results['T_c'] for gal in range(len(self.galaxies))])
        return t

    # Get dust emissivity spectral indexes of all galaxies
    def get_betas(self, fixed_beta=None, beta_inputs=None):
        """
        Obtains dust emissivity spectral indexes of galaxies
        
        :param fixed_beta: Fixed dust emissivity index for all galaxies (optional)
        :param beta_inputs: Set dust emissivity index for all galaxies (optional)
        :return: All dust emissivity spectral indexes in survey
        """
        if fixed_beta is not None:
            beta = np.array([fixed_beta for _ in range(len(self.galaxies))])
        elif beta_inputs is not None:
            beta = np.array(beta_inputs)
        else:
            beta = np.array([self.galaxies[gal].sed_results['beta'] for gal in range(len(self.galaxies))])
        return beta

    # Get monochromatic luminosities of all galaxies
    def get_luminosities(self, wave, fixed_t=None, fixed_beta=None, t_inputs=None, beta_inputs=None, redshift_inputs=None, progress=True):
        """
        Obtains monochromatic luminosities of galaxies
        
        :param wave: Wavelength of monochromatic luminosity [microns]
        :param fixed_t: Fixed dust temperature for all galaxies (optional)
        :param fixed_beta: Fixed dust emissivity index for all galaxies (optional)
        :param t_inputs: Set dust temperatures for all galaxies (optional)
        :param beta_inputs: Set dust emissivity index for all galaxies (optional)
        :param redshift_inputs: Set redshifts for all galaxies (optional)
        :param progress: Show progress (Default True)
        :return: All monochromatic luminosities in survey
        """
        fluxes = self.get_fluxes(wave)
        if redshift_inputs is not None:
            redshifts = redshift_inputs
        else:
            redshifts = self.get_redshifts(randomized_z=False)
        temperatures = self.get_dust_temperatures(fixed_t, t_inputs)
        betas = self.get_betas(fixed_beta, beta_inputs)
        if progress:
            print('Calculating Monochromatic Luminosities')
        L = np.log10(utils.flux_to_luminosity(fluxes, wave, redshifts, temperatures, betas, self.parameters['cosmology']).value)
        return L

    # Get dust masses of all galaxies
    def get_dust_masses(self, wave, kappa, fixed_t=None, fixed_beta=None, t_inputs=None, beta_inputs=None, redshift_inputs=None, progress=True):
        """
        Obtains dust masses of galaxies
        
        :param wave: Wavelength of dust mass [microns]
        :param kappa: Dust mass absorption coefficient [m2 kg-1]
        :param fixed_t: Fixed dust temperature for all galaxies (optional)
        :param fixed_beta: Fixed dust emissivity index for all galaxies (optional)
        :param t_inputs: Set dust temperatures for all galaxies (optional)
        :param beta_inputs: Set dust emissivity index for all galaxies (optional)
        :param redshift_inputs: Set redshifts for all galaxies (optional)
        :param progress: Show progress (Default True)
        :return: All dust masses in survey
        """
        fluxes = self.get_fluxes(wave)
        if redshift_inputs is not None:
            redshifts = redshift_inputs
        else:
            redshifts = self.get_redshifts(randomized_z=False)
        temperatures = self.get_dust_temperatures(fixed_t, t_inputs)
        betas = self.get_betas(fixed_beta, beta_inputs)
        if progress:
            print('Calculating Dust Masses')
        M = np.log10(utils.flux_to_dust_mass(fluxes, wave, wave, redshifts, temperatures, betas, kappa, self.parameters['cosmology']).value)
        return M

    # Get correction factors of all galaxies
    def get_corrections(self, wave, redshift_inputs=None, progress=True):
        """
        Obtains correction factors of galaxies
        
        :param wave: Wavelength of function [microns]
        :param redshift_inputs: Set redshifts for all galaxies (optional)
        :param progress: Show progress (Default True)
        """
        c_total = np.ones_like(self.galaxies)

        # Sub-mm completeness interpolation
        if "submm correction" in self.parameters:
            if progress:
                print('Calculating Sub-mm Correction Factors')
            fluxes = self.get_fluxes(wave)
            s_comp_bins, cs_bins = self.parameters['submm correction']
            f_s = interp1d(s_comp_bins, cs_bins, bounds_error=False, fill_value="extrapolate")
            c_s = f_s(fluxes)
            c_total *= c_s

        # ID completeness interpolation
        if "id correction" in self.parameters:
            if progress:
                print('Calculating ID Correction Factors')
            if redshift_inputs is not None:
                redshifts = redshift_inputs
            else:
                redshifts = self.get_redshifts(randomized_z=False)
            z_comp_bins, cz_bins = self.parameters['id correction']
            f_z = interp1d(z_comp_bins, cz_bins, bounds_error=False, fill_value="extrapolate")
            c_z = f_z(redshifts)
            c_total *= c_z
        return c_total

    # Run the binned function
    def binned_function(self, model, method, wave, z_bins, logx_bins, fixed_t=None, fixed_beta=None, t_inputs=None, beta_inputs=None, fixed_limit=None, fixed_snr=None, kappa=None, errors=True, n_mc=100):
        """
        Derives a binned function for any galaxy parameter
        
        :param model: Choice of "luminosity" or "dust mass"
        :param method: Choice of "Vmax", "PC00" or "PC00+D11"
        :param wave: Target wavelength of function [microns]
        :param z_bins: Redshift bins
        :param logx_bins: Measured parameter bins [log(x)]
        :param fixed_t: Fixed dust temperature for all galaxies (optional)
        :param fixed_beta: Fixed dust emissivity index for all galaxies (optional)
        :param t_inputs: Set dust temperatures for all galaxies (optional)
        :param beta_inputs: Set dust emissivity index for all galaxies (optional)
        :param fixed_limit: Fixed flux limit for all galaxies (optional)
        :param fixed_snr: Fixed SNR limit for all galaxies (optional)
        :param kappa: Dust mass absorption coefficient [m2 kg-1] (optional)
        :param errors: Run Monte Carlo simulation for errors (Default True)
        :param n_mc: Number of Monte Carlo iterations (Default 10)
        :return: Binned function with phi and error on phi grids
        """

        # Determine if the function is a flux or SNR limited survey
        if fixed_limit:
            idx = np.array([self.galaxies[gal].flux_jy[self.galaxies[gal].wavelengths_obs_um == wave].item() for gal in range(len(self.galaxies))]) > fixed_limit
        elif fixed_snr:
            idx = np.array([self.galaxies[gal].flux_jy[self.galaxies[gal].wavelengths_obs_um == wave].item()/self.galaxies[gal].flux_error_jy[self.galaxies[gal].wavelengths_obs_um == wave].item() for gal in range(len(self.galaxies))]) > fixed_snr
        else:
            idx = np.arange(0, len(self.galaxies))

        # Obtain properties of all galaxies
        z_values = self.get_redshifts(randomized_z=False)[idx]
        c_values = self.get_corrections(wave)[idx]
        t_values = self.get_dust_temperatures(fixed_t, t_inputs)[idx]
        beta_values = self.get_betas(fixed_beta, beta_inputs)[idx]
        flux_lim_values = self.get_flux_limits(wave, fixed_limit=fixed_limit, fixed_snr=fixed_snr)[idx]

        # Define logx values based on chosen model
        if model == 'luminosity':
            logx_values = self.get_luminosities(wave, fixed_t, fixed_beta, t_inputs, beta_inputs, progress=True)[idx]
        elif model == 'dust mass':
            logx_values = self.get_dust_masses(wave, kappa, fixed_t, fixed_beta, t_inputs, beta_inputs, progress=True)[idx]
        else:
            raise ValueError('The first argument, "model", must be one of "luminosity" or "dust mass".')

        # Run function to get N,C,V,phi grids
        function_results = get_function(model, method, self.parameters, wave, z_bins, logx_bins, z_values, logx_values, c_values, t_values, beta_values, flux_lim_values, kappa=kappa, disable=False)

        # If errors, run Monte Carlo simulation
        if errors:
            min_z, max_z, n_z = min(z_bins), max(z_bins), len(z_bins) - 1
            min_logx, max_logx, n_logx = min(logx_bins), max(logx_bins), len(logx_bins) - 1
            z_bin_centers = (z_bins[:-1] + z_bins[1:]) / 2
            logx_bin_centers = (logx_bins[:-1] + logx_bins[1:]) / 2
            dlogx = (max_logx - min_logx) / n_logx

            # Poissonian errors from root of source counts
            phi_sigma_n = (np.sqrt(function_results['C_grid']) / function_results['V_grid']) / dlogx
            function_results['phi_sigma_n'] = phi_sigma_n

            # For photo-z error we randomize the redshifts
            phi_array = []
            for _ in tqdm(range(n_mc), desc='Monte Carlo Simulation'):

                # Select random redshifts and get correction factors
                z_values_it = self.get_redshifts(randomized_z=True)
                c_values_it = self.get_corrections(wave, redshift_inputs=z_values_it, progress=False)
                z_values_it_selected = z_values_it[idx]
                c_values_it_selected = c_values_it[idx]

                # Recalculate logx based on new redshifts
                if model == 'luminosity':
                    logx_values_it = self.get_luminosities(wave, fixed_t, fixed_beta, t_inputs, beta_inputs, redshift_inputs=z_values_it, progress=False)
                elif model == 'dust mass':
                    logx_values_it = self.get_dust_masses(wave, kappa, fixed_t, fixed_beta, t_inputs, beta_inputs, redshift_inputs=z_values_it, progress=False)
                logx_values_it_selected = logx_values_it[idx]

                # Bin completeness values
                C_list = []
                for z_it in range(len(z_bin_centers)):
                    for logx_it in range(len(logx_bin_centers)):
                        C = [c for c, z, logx in zip(c_values_it_selected, z_values_it_selected, logx_values_it_selected) if (z_bins[z_it] < z < z_bins[z_it + 1]) & (logx_bins[logx_it] < logx < logx_bins[logx_it + 1])]
                        C_list.append(np.nansum(C))

                # Form grids from lists of C and phi
                C_grid = np.array(C_list).reshape(n_z, n_logx)
                phi_grid = (C_grid / function_results['V_grid']) / dlogx
                phi_array.append(phi_grid)

            # Add error grids to dictionary
            function_results['phi_sigma_z'] = np.std(phi_array, axis=0)
            function_results['phi_sigma'] = np.sqrt((function_results['phi_sigma_n'] ** 2) + (function_results['phi_sigma_z'] ** 2))
        return function_results

    # Fitting of a binned function
    def binned_function_fitting(self, model, function_results, fixed_params_list=None, fixed_values_list=None, nwalkers=50, niters=10000, sigma=1e-3, logx_min_list=None, logx_max_list=None, progress=True):
        """
        Fits Schechter functions to the binned function
        
        :param model: Choice of "luminosity" or "dust mass"
        :param function_results: Dictionary of results from function
        :param fixed_params_list: List of fixed Schechter parameters (optional)
        :param fixed_values_list: List of fixed Schechter values (optional)
        :param nwalkers: Number of walkers in MC (Default 50)
        :param niters: Number of MC iterations (Default 10000)
        :param sigma: Exploration parameter of MC (Default 0.001)
        :param logx_min_list: Minimum measured values at each redshift [log(x)]
        :param logx_max_list: Maximum measured values at each redshift [log(x)]
        :param progress: Show progress (Default True)
        :return Dictionary of Schechter parameters at each redshift
        """
        results = utils.mcmc_array(model, function_results, fixed_params_list=fixed_params_list, fixed_values_list=fixed_values_list, nwalkers=nwalkers, niters=niters, sigma=sigma, logx_min_list=logx_min_list, logx_max_list=logx_max_list, progress=progress)
        return results
