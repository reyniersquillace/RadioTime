import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pypulse as pp
import scipy.optimize as opt
import scipy.signal as sgn
import bisect

def gaussian(x, sig, mu):
    curve = np.exp(-0.5*((x - mu)/sig)**2)/(sig*np.sqrt(2*np.pi))
    curve /= max(curve)
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


def get_fwhm(xs, lag_sum):
    '''
    This routine fits a FWHM to the central portion of a 1D ACF curve. 
    
    '''
    search_range = np.where(np.abs(np.arange(len(lag_sum)) / len(lag_sum) - 0.5) < 0.1)[0]
    max_idx = np.where(lag_sum == max(lag_sum[search_range]))[0]

    if len(max_idx) > 1:
        max_idx = max_idx[0]
    
    minima = sgn.argrelmin(lag_sum)
    low, high = find_boundaries(minima[0], max_idx)
    lag_sum[max_idx] = (lag_sum[max_idx - 1] + lag_sum[max_idx + 1]) / 2
    lag_sum /= lag_sum[max_idx]
    sig, mu = opt.curve_fit(gaussian, xs[low:high + 1], lag_sum[low:high+1], 
                        [0.05*xs[-1], 0.5*xs[-1]])[0]
    
    fwhm_conv = 2*np.sqrt(2 * np.log(2))

    return fwhm_conv * sig, mu


def fit1D(ds, hrs, freqs):
    
    acf = ds.getACF()
    hrs = hrs[-1] - hrs[0]
    freqs = freqs[-1] - freqs[0]

    #sum over full acf for now
    time_lag = np.sum(acf, axis = 0)
    freq_lag = np.sum(acf, axis = 1)

    dt, t0 = get_fwhm(hrs, time_lag)
    dnu, nu0 = get_fwhm(freqs, freq_lag)



