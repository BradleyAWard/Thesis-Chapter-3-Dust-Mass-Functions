# --------------------------------------------------
# Imports
# --------------------------------------------------

import utils
import numpy as np
from scipy.special import gamma

# --------------------------------------------------
# Dust Mass Density (Literature)
# --------------------------------------------------

def dust_mass_density(study):
    """
    Determines the dust mass density for a literature study
    
    :param study: A dictionary for the study
    :return: The study with dust mass density included
    """
    # Locate the Mstar, phistar and alpha values in study
    m_star_low, m_star, m_star_high = 10**study['logm_star']-10**study['logm_star_low'], 10**study['logm_star'], 10**study['logm_star_high']-10**study['logm_star']
    phi_star_low, phi_star, phi_star_high = (10**study['logphi_star']-10**study['logphi_star_low']), (10**study['logphi_star']), (10**study['logphi_star_high']-10**study['logphi_star'])
    alpha = study['alpha']

    # Calculation of dust mass density and errors
    dmd = phi_star*m_star*gamma(alpha+2)
    study['dmd'] = dmd
    study['dmd_low'] = dmd-(dmd*np.sqrt(((phi_star_low/phi_star)**2) + ((m_star_low/m_star)**2)))
    study['dmd_high'] = dmd+(dmd*np.sqrt(((phi_star_high/phi_star)**2) + ((m_star_high/m_star)**2)))
    return study

# --------------------------------------------------

def dust_mass_density_literature(literature):
    """
    Creates a dictionary of studies with dust mass densities

    :param literature: A dictionary of studies
    :return: Literature studies with dust mass densities
    """
    literature_new = {}
    study_names = literature.keys()
    for study in study_names:
        literature_new[study] = dust_mass_density(literature[study])
    return literature_new

# --------------------------------------------------
# Conversion to common cosmology and kappa
# --------------------------------------------------

def convert_mass(log_mass, dL_old, dL_new, kappa_old, kappa_new):
    """
    Converts a dust mass to a common cosmology and kappa
    
    :param log_mass: Dust mass [log(Msun)]
    :param dL_old: Luminosity distance (original cosmology) [Mpc]
    :param dL_new: Luminosity distance (new cosmology) [Mpc]
    :param kappa_old: Old dust mass absorption coefficient [m2 kg-1]
    :param kappa_new: New dust mass absorption coefficient [m2 kg-1]
    :return: Dust mass scaled to cosmology and kappa [log(Msun)]
    """
    mass = 10**log_mass
    mass_new = mass*((dL_new**2)/kappa_new)*(kappa_old/(dL_old**2))
    log_mass_new = np.log10(mass_new)
    return log_mass_new

# --------------------------------------------------

def convert_normalization(log_phi, v_old, v_new):
    """
    Converts a normalization to a common cosmology and kappa
    
    :param log_phi: Normalization of dust mass function [log(Mpc-3)]
    :param v_old: Comoving volume (original cosmology) [Mpc3]
    :param v_new: Comoving volume (new cosmology) [Mpc3]
    :return: Normaliztion scaled to cosmology [log(Mpc-3)]
    """
    phi = 10**log_phi
    phi_new = phi*(v_old/v_new)
    log_phi_new = np.log10(phi_new)
    return log_phi_new

# --------------------------------------------------

def convert_study(study, cosmology_new, kappa_new):
    """
    Converts a study to common cosmology and kappa
    
    :param study: A study with known DMF parameters
    :param cosmology_new: Chosen cosmological parameters
    :param kappa_new: Chosen dust mass absorption coefficient
    :return: List of new scaled DMF parameters (yes I know its ugly)
    """
    # Extract kappa and cosmological parameters
    kappa_old = study['kappa']
    H0_old, Om_old, OK_old, Olam_old = study['cosmology']
    H0_new, Om_new, OK_new, Olam_new = cosmology_new

    # Rescale mass given M_star is proportional to Dl^2/kappa
    z_test = 0.1
    dl_old = utils.dL(z_test, H0_old, Om_old, OK_old, Olam_old)
    dl_new = utils.dL(z_test, H0_new, Om_new, OK_new, Olam_new)
    logm_star_low_old, logm_star_old, logm_star_high_old = study['logm_star_low'], study['logm_star'], study['logm_star_high']
    logm_star_low_new = convert_mass(logm_star_low_old, dl_old, dl_new, kappa_old, kappa_new)
    logm_star_new = convert_mass(logm_star_old, dl_old, dl_new, kappa_old, kappa_new)
    logm_star_high_new = convert_mass(logm_star_high_old, dl_old, dl_new, kappa_old, kappa_new)

    # Rescale normalization given phi_star is proportional to 1/V
    v_old = utils.differential_comoving_v(z_test, H0_old, Om_old, OK_old, Olam_old)
    v_new = utils.differential_comoving_v(z_test, H0_new, Om_new, OK_new, Olam_new)
    logphi_star_perdex_low_old, logphi_star_perdex_old, logphi_star_perdex_high_old = study['logphi_star_perdex_low'], study['logphi_star_perdex'], study['logphi_star_perdex_high']
    logphi_star_low_old, logphi_star_old, logphi_star_high_old = np.log10((10**logphi_star_perdex_low_old)/np.log(10)), np.log10((10**logphi_star_perdex_old)/np.log(10)), np.log10((10**logphi_star_perdex_high_old)/np.log(10))
    logphi_star_low_new = convert_normalization(logphi_star_low_old, v_old, v_new)
    logphi_star_new = convert_normalization(logphi_star_old, v_old, v_new)
    logphi_star_high_new = convert_normalization(logphi_star_high_old, v_old, v_new)
    logphi_star_perdex_low_new, logphi_star_perdex_new, logphi_star_perdex_high_new = np.log10((10**logphi_star_low_new)*np.log(10)), np.log10((10**logphi_star_new)*np.log(10)), np.log10((10**logphi_star_high_new)*np.log(10))
    return logm_star_low_new, logm_star_new, logm_star_high_new, logphi_star_low_new, logphi_star_new, logphi_star_high_new, logphi_star_perdex_low_new, logphi_star_perdex_new, logphi_star_perdex_high_new

# --------------------------------------------------

def convert_literature(literature, cosmology_new, kappa_new):
    """
    Converts a set of studies to a common cosmology and kappa
    
    :param literature: Dictionary of studies with DMF parameters
    :param cosmology_new: Chosen cosmological parameters
    :param kappa_new: Chosen dust mass absorption coefficient
    :return: Dictionary of studies with rescaled DMF parameters
    """
    literature_converted = {}
    literature_names = list(literature.keys())
    for study in literature_names:

        # Set up a new dictionary with default values
        study_values = {}
        study_values.setdefault('z_bins', literature[study]['z_bins'])
        study_values.setdefault('z_bin_centers', literature[study]['z_bin_centers'])
        study_values.setdefault('logm_range', literature[study]['logm_range'])
        study_values.setdefault('alpha_low', literature[study]['alpha_low'])
        study_values.setdefault('alpha', literature[study]['alpha'])
        study_values.setdefault('alpha_high', literature[study]['alpha_high'])

        # Gather the new converted values
        logm_star_low_new, logm_star_new, logm_star_high_new, logphi_star_low_new, logphi_star_new, logphi_star_high_new, logphi_star_perdex_low_new, logphi_star_perdex_new, logphi_star_perdex_high_new = convert_study(literature[study], cosmology_new, kappa_new)

        # Replace the default values if available
        study_values.setdefault('logm_star_low', logm_star_low_new)
        study_values.setdefault('logm_star', logm_star_new)
        study_values.setdefault('logm_star_high', logm_star_high_new)
        study_values.setdefault('logphi_star_perdex_low', logphi_star_perdex_low_new)
        study_values.setdefault('logphi_star_perdex', logphi_star_perdex_new)
        study_values.setdefault('logphi_star_perdex_high', logphi_star_perdex_high_new)
        study_values.setdefault('logphi_star_low', logphi_star_low_new)
        study_values.setdefault('logphi_star', logphi_star_new)
        study_values.setdefault('logphi_star_high', logphi_star_high_new)
        literature_converted[study] = study_values
    return literature_converted
