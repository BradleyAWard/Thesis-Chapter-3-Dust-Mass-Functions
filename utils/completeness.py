# --------------------------------------------------
# Imports
# --------------------------------------------------

import numpy as np
from astropy import units as u

# --------------------------------------------------
# ID Completeness Function
# --------------------------------------------------

def id_completeness(sources, background, reliables, N_sources, N_background, z_string, n_z_bins, z_range, Q, Q_err, r):
    """
    Determines the completeness of the IDs as a function of z
    
    :param sources: Dataframe of sources
    :param background: Dataframe of background sources
    :param reliables: Dataframe of reliable IDs
    :param N_sources: The number of sources
    :param N_background: The number of background sources
    :param z_string: Name of the redshift column for dataframes
    :param n_z_bins: Number of redshift bins for function
    :param z_range: The minimum and maximum redshift values
    :param Q: Value of Q (fraction of sources with observable candidates)
    :param Q_err: Error on Q (fraction of sources with candidates)
    :param r: The maximum search radius [arcsec]
    :return: The ID completeness as a function of redshift
    """
    # Define histograms of sources, background sources and reliably matched sources
    n_data, z_bins = np.histogram(sources[z_string], bins=n_z_bins, range = (z_range[0], z_range[1]))
    n_back, _ = np.histogram(background[z_string], bins=n_z_bins, range = (z_range[0], z_range[1]))
    n_reliable, _ = np.histogram(reliables[z_string], bins=n_z_bins, range = (z_range[0], z_range[1]))
    z_bin_centers = (z_bins[:-1] + z_bins[1:])/2

    # Normalize the source and background histograms
    n_data_norm = n_data/(N_sources*np.pi*((r*u.arcsec)**2))
    n_back_norm = n_back/(N_background*np.pi*((r*u.arcsec)**2))

    # Calculate the true counterparts distribution as a function of redshift
    n_real = n_data_norm - n_back_norm
    q = (n_real*Q)/sum(n_real)

    # Determine the completeness and correction factors
    comp = n_reliable/(q*N_sources)
    c_factor = 1/comp

    # Calculate the errors from dataframes
    n_data_norm_err = np.sqrt(n_data)/(N_sources*np.pi*((r*u.arcsec)**2))
    n_back_norm_err = np.sqrt(n_back)/(N_background*np.pi*((r*u.arcsec)**2))
    n_reliable_err = np.sqrt(n_reliable)

    # Propagate errors to the n(real) distribution and completeness
    n_real_err = np.sqrt((n_data_norm_err**2) + (n_back_norm_err**2))
    n_real_sum_err = np.sqrt(sum(n_real_err**2))
    q_err = np.sqrt(((n_real_err/n_real)**2) + ((Q_err/Q)**2) + ((n_real_sum_err/sum(n_real))**2))*q
    comp_err = np.sqrt(((n_reliable_err/n_reliable)**2) + ((q_err/q)**2))*comp
    c_factor_err = abs(-1)*(comp_err/comp)*c_factor

    return z_bins, z_bin_centers, comp, comp_err, c_factor, c_factor_err