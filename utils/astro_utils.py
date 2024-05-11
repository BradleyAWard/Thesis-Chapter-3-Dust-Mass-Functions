# --------------------------------------------------
# Imports
# --------------------------------------------------

import numpy as np
from astropy import units as u
from scipy.integrate import quad
from astropy.constants import c, h, k_B, M_sun

# --------------------------------------------------
# Cosmological Functions
# --------------------------------------------------

def dH(H0):
    """
    Returns the Hubble distance

    :param H0: Hubble parameter
    :return: Hubble distance [Mpc]
    """
    return (c/H0).to(u.Mpc)

# --------------------------------------------------

def E(z, Om, Ok, Olam):
    """
    Returns the dimensionless Hubble parameter
    
    :param z: Redshift
    :param Om: Matter density parameter
    :param Ok: Curvature density parameter
    :param Olam: Dark energy density parameter
    :return: Dimensionless Hubble parameter
    """
    return np.sqrt((Om*((1+z)**3)) + (Ok*((1+z)**2)) + Olam)

# --------------------------------------------------

def dC(z, H0, Om, Ok, Olam):
    """
    Returns the comoving distance
    
    :param z: Redshift
    :param H0: Hubble parameter
    :param Om: Matter density parameter
    :param Ok: Curvature density parameter
    :param Olam: Dark energy density parameter
    :return: Comvoing distance [Mpc]
    """
    DH = dH(H0)
    def integrand(z):
        return 1/E(z, Om, Ok, Olam)
    integration = np.vectorize(lambda z: quad(integrand, 0, z))
    integral_result, _ = integration(z)
    return DH*integral_result

# --------------------------------------------------

def dM(z, H0, Om, Ok, Olam):
    """
    Returns the transverse comoving distance
    
    :param z: Redshift
    :param H0: Hubble parameter
    :param Om: Matter density parameter
    :param Ok: Curvature density parameter
    :param Olam: Dark energy density parameter
    :return: Transverse comvoing distance [Mpc]
    """
    DH = dH(H0)
    DC = dC(z, H0, Om, Ok, Olam)
    sinh_func = (np.sqrt(np.abs(Ok))*DC)/DH
    if Ok > 0:
        return (DH*np.sinh(sinh_func))/np.sqrt(Ok)
    elif Ok == 0:
        return DC
    elif Ok < 0:
        return (DH*np.sin(sinh_func))/np.sqrt(np.abs(Ok))

# --------------------------------------------------

def dA(z, H0, Om, Ok, Olam):
    """
    Returns the angular diameter distance
    
    :param z: Redshift
    :param H0: Hubble parameter
    :param Om: Matter density parameter
    :param Ok: Curvature density parameter
    :param Olam: Dark energy density parameter
    :return: Angular diameter distance [Mpc]
    """
    DM = dM(z, H0, Om, Ok, Olam)
    return DM/(1+z)

# --------------------------------------------------

def dL(z, H0, Om, Ok, Olam):
    """
    Returns the luminosity distance
    
    :param z: Redshift
    :param H0: Hubble parameter
    :param Om: Matter density parameter
    :param Ok: Curvature density parameter
    :param Olam: Dark energy density parameter
    :return: Luminosity distance [Mpc]
    """
    DM = dM(z, H0, Om, Ok, Olam)
    return DM*(1+z)

# --------------------------------------------------

def differential_comoving_v(z, H0, Om, Ok, Olam):
    """
    Returns the differential comoving volume
    
    :param z: Redshift
    :param H0: Hubble parameter
    :param Om: Matter density parameter
    :param Ok: Curvature density parameter
    :param Olam: Dark energy density parameter
    :return: Differential comoving volume [Mpc3]
    """
    DH = dH(H0)
    DA = dA(z, H0, Om, Ok, Olam)
    E_z = E(z, Om, Ok, Olam)
    return ((DH*((1+z)**2)*(DA**2))/E_z)/u.sr

# --------------------------------------------------
# Galaxy Properties
# --------------------------------------------------

def blackbody(lam_um, t):
    """
    A blackbody function
    
    :param lam_um: Wavelength [microns]
    :param t: Dust temperature [K]
    :return: Blackbody function
    """
    lam_m = (lam_um*u.micron).to(u.m)
    nu = c/lam_m
    bb = ((2*h*(nu**3))/(c**2))*(1/(np.exp((h*nu)/(k_B*t))-1))
    return bb

# --------------------------------------------------

def flux_to_luminosity(f, wave_obs_um, z, T, beta, cosmology):
    """
    Converts a flux density to a monochromatic luminosity
    
    :param f: Flux density [Jy]
    :param wave_obs_um: Observed wavelength [microns]
    :param z: Redshift
    :param T: Dust temperature [K]
    :param beta: Dust emissivity spectral index
    :param cosmology: List of cosmological parameters
    :return: Monochromatic luminosity [W Hz-1]
    """
    # Definitions
    f = f*u.Jy
    T = T*u.K
    wave_obs_m = (wave_obs_um*u.micron).to(u.m)
    nu_obs = c/wave_obs_m
    nu_rest = nu_obs*(1+z)

    # Calculation of luminosity from flux
    k = ((nu_obs/nu_rest)**(3+beta))*((np.exp((h*nu_rest)/(k_B*T))-1)/(np.exp((h*nu_obs)/(k_B*T))-1))
    H0, Om, Ok, Olam = cosmology
    DL = dL(z, H0, Om, Ok, Olam)
    L = (4*np.pi*(DL**2)*f*k)/(1+z)
    L = L.to(u.Watt/u.Hz)
    return L

# --------------------------------------------------

def luminosity_to_flux(L, wave_obs_um, z, T, beta, cosmology):
    """
    Converts a monochromatic luminosity to flux density
    
    :param L: Monochromatic luminosity [W Hz-1]
    :param wave_obs_um: Observed wavelength [microns]
    :param z: Redshift
    :param T: Dust temperature [K]
    :param beta: Dust emissivity spectral index
    :param cosmology: List of cosmological parameters
    :return: Flux density [Jy]
    """
    # Definitions
    L = L*(u.Watt/u.Hz)
    T = T*u.K
    wave_obs_m = (wave_obs_um*u.micron).to(u.m)
    nu_obs = c/wave_obs_m
    nu_rest = nu_obs*(1+z)

    # Calculation of flux from luminosity
    k = ((nu_obs/nu_rest)**(3+beta))*((np.exp((h*nu_rest)/(k_B*T))-1)/(np.exp((h*nu_obs)/(k_B*T))-1))
    H0, Om, Ok, Olam = cosmology
    DL = dL(z, H0, Om, Ok, Olam)
    f = (L*(1+z))/(4*np.pi*(DL**2)*k)
    f = f.to(u.Jy)
    return f

# --------------------------------------------------

def luminosity_to_dust_mass(L, wave_um, T, kappa):
    """
    Converts a monochromatic luminosity to dust mass
    
    :param L: Monochromatic luminosity [W Hz-1]
    :param wave_um: Wavelength [microns]
    :param T: Dust temperature [K]
    :param kappa: Dust absorption coefficient [m2 kg-1]
    :return: Dust mass [Msun]
    """
    # Definitions
    L = L*(u.Watt/u.Hz)
    kappa = kappa*(u.m**2/u.kg)
    T = T*u.K
    bb = blackbody(wave_um, T).to(u.Jy)

    # Calculation of dust mass from luminosity
    dust_mass = L/(4*np.pi*bb*kappa)
    dust_mass_kg = dust_mass.to(u.kg)
    dust_mass_solar = dust_mass_kg/M_sun
    return dust_mass_solar

# --------------------------------------------------

def dust_mass_to_luminosity(M, wave_um, T, kappa):
    """
    Converts a dust mass to monochromatic luminosity
    
    :param M: Dust mass [Msun]
    :param wave_um: Wavelength [microns]
    :param T: Dust temperature [K]
    :param kappa: Dust absorption coefficient [m2 kg-1]
    :return: Monochromatic luminosity [W Hz-1]
    """
    # Definitions
    dust_mass_kg = M*M_sun
    kappa = kappa*(u.m**2/u.kg)
    T = T*u.K
    bb = blackbody(wave_um, T).to(u.Jy)

    # Calculation of luminosity from dust mass
    L = 4*np.pi*bb*kappa*dust_mass_kg
    L = L.to(u.Watt/u.Hz)
    return L

# --------------------------------------------------

def flux_to_dust_mass(f, wave_obs_um, wave_um, z, T, beta, kappa, cosmology):
    """
    Converts a flux density to dust mass
    
    :param f: Flux density [Jy]
    :param wave_obs_um: Observed wavelength [microns]
    :param wave_um: Target wavelength [microns]
    :param z: Redshift
    :param T: Dust temperature [K]
    :param beta: Dust emissivity spectral index
    :param kappa: Dust absorption coefficient [m2 kg-1]
    :param cosmology: List of cosmological parameters
    :return: Dust mass [Msun]
    """
    L = flux_to_luminosity(f, wave_obs_um, z, T, beta, cosmology)
    L = L.value
    M = luminosity_to_dust_mass(L, wave_um, T, kappa)
    return M

# --------------------------------------------------

def dust_mass_to_flux(M, wave_obs_um, wave_um, z, T, beta, kappa, cosmology):
    """
    Converts a dust mass to flux density
    
    :param M: Dust mass [Msun]
    :param wave_obs_um: Observed wavelength [microns]
    :param wave_um: Target wavelength [microns]
    :param z: Redshift
    :param T: Dust temperature [K]
    :param beta: Dust emissivity spectral index
    :param kappa: Dust absorption coefficient [m2 kg-1]
    :param cosmology: List of cosmological parameters
    :return: Flux density [Jy]
    """
    L = dust_mass_to_luminosity(M, wave_um, T, kappa)
    L = L.value
    f = luminosity_to_flux(L, wave_obs_um, z, T, beta, cosmology)
    return f

# --------------------------------------------------

def convert_observed_fluxes(f_wave_old, wave_um_old, wave_um_new, T, beta):
    """
    Applies a K-correction to fluxes
    
    :param f_wave_old: Flux density at original wavelength [Jy]:param wave_um_old: Original wavelength [microns]
    :param wave_um_new: New wavelength [microns]
    :param T: Dust temperature [K]
    :param beta: Dust emissivity spectral index
    :return: K-corrected flux density
    """
    # Definitions
    T = T*u.K
    wave_m_old = (wave_um_old*u.micron).to(u.m)
    wave_m_new = (wave_um_new*u.micron).to(u.m)
    nu_old = c/wave_m_old
    nu_new = c/wave_m_new

    # Calculation of K-correction by ratio of fluxes
    f_wave_new = f_wave_old*((nu_new/nu_old)**(3+beta))*((np.exp((h*nu_old)/(k_B*T))-1)/(np.exp((h*nu_new)/(k_B*T))-1))
    return f_wave_new
