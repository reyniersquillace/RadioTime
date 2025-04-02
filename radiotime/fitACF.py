import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pypulse as pp
import scipy.optimize as opt
import scipy.signal as sgn
import bisect
import astropy.units as u

def gaussian(x, sig, mu, shift, scale):

    curve = np.exp(-0.5*((x - mu)/sig)**2)/(sig*np.sqrt(2*np.pi))
    curve *= scale / max(curve)
    curve += shift

    return curve

def lorentzian(x, fwhm, mu, shift, scale):
     
    hwhm = fwhm / 2

    curve = (hwhm / ((x - mu)**2 + hwhm**2 )) / np.pi
    curve *= scale / max(curve)
    curve += shift

    return curve

def find_boundaries(A, B):
    '''
    This function takes a sorted array A and finds the two values
    that sandiwch B.

    Inputs:
    -------
        A (arr)        : a sorted array from least to greatest
        B (int / float): the value that you want to insert

    Returns:
        The two values in A on either side of B.
    '''

    #find where to insert B into A
    idx = bisect.bisect_left(A, B)
    #check there is a place for the peak value to be inserted
    if idx == 0 or idx == len(A):
        print("There's something really weird with this observation.")
    else:
        #B is between A[idx-1] and A[idx]
        return A[idx-1], A[idx]
    
def get_range(lag_sum, order = 10):
     
    search_range = np.where(np.abs(np.arange(len(lag_sum)) / \
                                   len(lag_sum) - 0.5) < 0.1)[0]
    max_idx = np.where(lag_sum == max(lag_sum[search_range]))[0]

    if len(max_idx) > 1:
        max_idx = max_idx[0]
    
    minima = sgn.argrelmin(lag_sum, order = order)
    low, high = find_boundaries(minima[0], max_idx)
    #make the upper limit inclusive
    high += 1

    return low, high, max_idx

def get_stamp(acf, time_lag, freq_lag, plot = True):
    ''' 
    This function finds the "stamp" of the ACF and recalculates each 

    '''

    time_acf = np.sum(acf, axis = 0)
    freq_acf = np.sum(acf, axis = 1)

    tlow, thigh, tmax = get_range(time_acf)
    flow, fhigh, fmax = get_range(freq_acf)

    time_acf = np.sum(acf[flow:fhigh], axis = 0)
    freq_acf = np.sum(acf[:, tlow:thigh], axis = 1)
    time_lags = np.linspace(-time_lag, time_lag, len(time_acf))
    freq_lags = np.linspace(-freq_lag, freq_lag, len(freq_acf))

    if plot:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
        ax1.imshow(acf,
                    aspect = 'auto',
                    cmap = 'magma',
                    extent = [-time_lag, time_lag,
                              -freq_lag, freq_lag],           
        )
        ax2.imshow(acf[flow:fhigh, tlow:thigh],
                    aspect = 0.05,
                    cmap = 'magma',
                    extent = [time_lags[tlow], time_lags[thigh],
                              freq_lags[flow], freq_lags[fhigh]],
         )
        ax1.set_title('Full ACF')
        ax1.set_xlabel('Time Lag (hr)')
        ax1.set_ylabel('Frequency Lag (MHz)')
        ax2.set_title('Fitting Region')
        ax2.set_xlabel('Time Lag (hr)')
        plt.tight_layout()

    return time_acf, time_lags, freq_acf, freq_lags

def get_fwhm(xs, lag_sum, plot = True, lagtype = None):
    '''
    This routine fits a FWHM to the central portion of a 1D ACF curve. 

    '''

    if len(xs) != len(lag_sum):
        raise ValueError(f"Arrays must have the same length. \n\
            Got lengths {len(xs)} and {len(lag_sum)}.")
        return
    
    low, high, max_idx = get_range(lag_sum)
    lag_sum[max_idx] = (lag_sum[max_idx - 1] + lag_sum[max_idx + 1]) / 2
    lag_sum /= lag_sum[max_idx]

    par_gauss = opt.curve_fit(gaussian, xs[low:high], lag_sum[low:high], 
                        p0 = [0.02*xs[high], 0, 0, 1])
    sig, mu_gauss, shift_gauss, scale_gauss = par_gauss[0]
    cov_gauss = par_gauss[1]
    fwhm_conv = 2*np.sqrt(2 * np.log(2))
    fwhm_gauss = fwhm_conv * sig
    # #there is no way this is legal:
    # perr_gauss = np.sqrt(np.sum(np.diag(cov_gauss)**2))

    # par_lorentz = opt.curve_fit(lorentzian, xs[low:high], lag_sum[low:high], 
    #                                     p0 = [0.02*xs[high], 0, 0.5, 0.5])
    # fwhm_lorentz, mu_lorentz, shift_lorentz, scale_lorentz = par_lorentz[0]
    # cov_lorentz = par_lorentz[1]
    # #what the fuck am i doing here
    # perr_lorentz = np.sqrt(np.sum(np.diag(cov_lorentz))**2)

    if plot:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
        ax1.set_title('Full ACF')
        ax1.plot(xs, lag_sum, 'k')
        ax1.plot(xs[low:high], lag_sum[low:high], 'r', label = 'Fit Range')
        ax1.plot(xs[low:high], gaussian(xs[low:high], sig, mu_gauss, shift_gauss, 
                scale_gauss), 'b--', label = 'Gaussian Fit')
        #ax1.plot(xs[low:high], lorentzian(xs[low:high], fwhm_lorentz, mu_lorentz, 
        #        shift_gauss, scale_gauss), 'g--', label = 'Lorentzian Fit')
        
        ax2.set_title('Central Region')
        ax2.plot(xs[low:high], lag_sum[low:high], 'r', label = 'Fit Range')
        ax2.plot(xs[low:high], gaussian(xs[low:high], sig, mu_gauss, shift_gauss,
                scale_gauss), 'b--', 
                label = 'Gaussian Fit')
        #ax2.plot(xs[low:high], lorentzian(xs[low:high], fwhm_lorentz, 
        #        mu_lorentz, shift_lorentz, scale_lorentz), 'g--', 
        #        label = 'Lorentzian Fit')

        if lagtype in ('time', 'Time', 't'):
                fig.suptitle('1D Time ACF')
                ax1.set_xlabel('Time Lag (hrs)')
                ax2.set_xlabel('Time Lag (hrs)')

        elif lagtype in ('Frequency', 'freq', 'frequency', 'f'):
                fig.suptitle('1D Frequency ACF')
                ax1.set_xlabel('Frequency Lag (MHz)')
                ax2.set_xlabel('Frequency Lag (MHz)')

        ax1.set_ylabel('Intensity')
        ax1.legend()
        ax2.legend()

    # if perr_gauss < perr_lorentz:   
    #     print(f'Best fit for {lagtype} was a Gaussian.')
    #     return fwhm_gauss, mu_gauss
    # else:
    #     print(f'Best fit for {lagtype} was a Lorentzian.')
    #     return fwhm_lorentz, mu_lorentz
    
    return fwhm_gauss, mu_gauss


def fit1D(ds, hrs, freqs):
    
    acf = ds.getACF()
    time_lag = hrs[-1] - hrs[0]
    freq_lag = freqs[-1] - freqs[0]
    time_acf, time_lags, freq_acf, freq_lags = get_stamp(acf, time_lag, freq_lag)

    dt, t0 = get_fwhm(time_lags, time_acf, lagtype = 'time') * u.hr
    dnu, nu0 = get_fwhm(freq_lags, freq_acf, lagtype = 'frequency') * u.MHz

    print(f'dt = {np.round(dt.to(u.min), 3)}, dnu = {np.round(dnu, 3)}')

    return dt, dnu

def scattering_timescale(ds, hrs, freqs, C1 = 1, eta_t = 0.2, eta_nu = 0.2):
     
    dt, dnu = fit1D(ds, hrs, freqs)
    tau_d = (C1 / 2 / np.pi / dnu).to(u.us)
    print(f'tau_d = {np.round(tau_d, 3)}')
    T = (hrs[-1] - hrs[0])*u.hr
    B = ((freqs[-1] - freqs[0])*u.MHz).to(u.Hz)

    n_ISS = (1 + eta_t * T / dt) * (1 + eta_nu*B / dnu)
    sig_DISS = (tau_d / np.sqrt(n_ISS))

    return tau_d, sig_DISS



