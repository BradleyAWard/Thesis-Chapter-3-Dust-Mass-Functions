# --------------------------------------------------
# Imports
# --------------------------------------------------

import numpy as np
from astropy.constants import c, h, k_B
from lmfit import Model, Parameters
from tqdm import tqdm

# --------------------------------------------------
# SED Fitting 
# --------------------------------------------------

def one_temperature(lam_um_obs, z, log_N_c, T_c, beta):
    """
    Defines a one temperature SED
    
    :param lam_um_obs: Observed wavelength [microns]
    :param z: Redshift
    :param log_N_c: Cold dust normalization
    :param T_c: Cold dust temperature [K]
    :param beta: Dust emissivity spectral index
    :return: One temperature component SED
    """
    N_c = 10 ** log_N_c
    lam_um_rest = lam_um_obs/(1+z)
    lam_m_rest = lam_um_rest * 1e-6
    nu_rest = c.value / lam_m_rest
    bb_c = (nu_rest**3)/(np.exp((h.value * nu_rest) / (k_B.value * T_c)) - 1)
    return N_c*(nu_rest**beta)*bb_c

# --------------------------------------------------

def fit_one_temperature(wavelengths_um_obs, fluxes, flux_errors, z, fixed_t=None, t_range=None, fixed_beta=None, beta_range=None):
    """
    Fits a one temperature model to an SED
    
    :param wavelengths_um_obs: Observed wavelengths [microns]
    :param fluxes: Flux densities [Jy]
    :param flux_errors: Errors on flux densities [Jy]
    :param z: Redshift
    :param fixed_t: Fixed dust temperature (optional) [K]
    :param t_range: Range of possible dust temperatures (optional) [K]
    :param fixed_beta: Fixed dust emissivity spectral index (optional)
    :param beta_range: Range of possible dust emissivity index (optional)
    :return: Dictionary of best fitting SED parameters
    """

    # Choices allowed for the dust emissivity spectral index
    if fixed_beta is not None:
        beta_min, beta_max = fixed_beta-0.001, fixed_beta+0.001
        beta = fixed_beta
        beta_vary = False
    elif beta_range is not None:
        beta_min, beta_max = beta_range
        beta = (beta_min+beta_max)/2
        beta_vary = True
    else:
        beta_min, beta_max = 1, 5
        beta = 2
        beta_vary = True

    # Choices allowed for the dust temperature
    if fixed_t is not None:
        t_min, t_max = fixed_t-0.001, fixed_t+0.001
        t = fixed_t
        t_vary = False
    elif t_range is not None:
        t_min, t_max = t_range
        t = (t_min+t_max)/2
        t_vary = True
    else:
        t_min, t_max = 10, 50
        t = 20
        t_vary = True

    # Set up parameters 
    sed_one_temp = Model(one_temperature)
    params = Parameters()
    params.add_many(('z', z, False, 0.0001, 3),
                    ('log_N_c', -60., True, -63, -54),
                    ('T_c', t, t_vary, t_min, t_max),
                    ('beta', beta, beta_vary, beta_min, beta_max))

    # Gathering best fitting values from lmfit
    result = sed_one_temp.fit(fluxes, params=params, lam_um_obs=wavelengths_um_obs, weights=1/flux_errors)
    log_N_c, T_c, beta = result.params['log_N_c'].value, result.params['T_c'].value, result.params['beta'].value
    T_c_err = result.params['T_c'].stderr
    result = {'log_N_c': log_N_c, 'T_c': T_c, 'T_c_err': T_c_err, 'beta': beta}
    return result

# --------------------------------------------------
# Galaxy Class
# --------------------------------------------------

class Galaxy:
    """
    Class for a single galaxy
    """
    def __init__(self, id, redshift, redshift_err, wavelengths_obs_um, flux_jy, flux_error_jy, calibration_percent, sed_wavelengths_obs_um, fixed_t=None, t_range=None, fixed_beta=None, beta_range=None):
        self.id = id
        self.redshift = redshift
        self.redshift_err = redshift_err
        self.wavelengths_obs_um = wavelengths_obs_um
        self.flux_jy = flux_jy
        self.flux_error_jy = flux_error_jy
        self.calibration_percent = calibration_percent
        self.sed_wavelengths_obs_um = sed_wavelengths_obs_um

        # Determine calibration errors
        calibration_err = [(percent / 100) * flux for percent, flux in zip(self.calibration_percent, self.flux_jy)]
        self.fluxerr_jy = np.array([np.sqrt((flux_err ** 2) + (cal_err ** 2)) for flux_err, cal_err in zip(self.flux_error_jy, calibration_err)])

        # Add calibration errors to flux density errors
        self.sed_flux_jy = np.array([flux for wave, flux, flux_error in zip(self.wavelengths_obs_um, self.flux_jy, self.fluxerr_jy) if wave in self.sed_wavelengths_obs_um])
        self.sed_fluxerr_jy = np.array([flux_error for wave, flux, flux_error in zip(self.wavelengths_obs_um, self.flux_jy, self.fluxerr_jy) if wave in self.sed_wavelengths_obs_um])

        # SED Fitting
        self.sed_results = fit_one_temperature(self.sed_wavelengths_obs_um, self.sed_flux_jy, self.sed_fluxerr_jy, self.redshift, fixed_t, t_range, fixed_beta, beta_range)

# --------------------------------------------------
# Samples of galaxies
# --------------------------------------------------

def create_sample(data, data_params, fixed_t=None, t_range=None, fixed_beta=None, beta_range=None):
    """
    Creates a sample of galaxies
    
    :param data: Datafile for galaxies
    :param data_params: Dictionary containing column names for data
    :param fixed_t: Fixed dust temperature (optional) [K]
    :param t_range: Range of possible dust temperatures (optional) [K]
    :param fixed_beta: Fixed dust emissivity spectral index (optional)
    :param beta_range: Range of possible dust emissivity index (optional)
    :return: List of galaxies
    """
    galaxies = []
    for gal in tqdm(range(len(data)), desc='Creating Sample'):

        # Identify an ID, redshift and flux densities from data
        id = data[data_params['ID']][gal]
        z = data[data_params['redshift']][gal]
        if 'redshift_error' in data_params:
            z_err = data[data_params['redshift_error']][gal]
        else:
            z_err = 0
        flux = np.array(data[data_params['flux_jy']].iloc[gal])
        flux_err = np.array(data[data_params['flux_error_jy']].iloc[gal])

        # Create an instance of the galaxy class, run and add to list
        galaxy = Galaxy(id, z, z_err, data_params['wavelengths_obs_um'], flux, flux_err, data_params['calibration_percent'], data_params['sed_wavelengths_obs_um'], fixed_t, t_range, fixed_beta, beta_range)
        galaxies.append(galaxy)

    return galaxies
