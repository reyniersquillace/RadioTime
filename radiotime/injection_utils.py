import numpy as np
import scipy.stats as stats
from scipy.fft import fft
from scipy.special import chdtri

nchan = 16384  #number of spectral channels in backend -- CHIME
default_freqs = np.linspace(400.0, 800.0, nchan)  # in MHz for CHIME band
default_nbins = 1024 #for CHIME

'''
This cell was written by Scott Ransom.
'''
def fft_rotate(arr, bins):
    """
    fft_rotate(arr, bins):
        Return array 'arr' rotated by 'bins' places to the left.  The
            rotation is done in the Fourier domain using the Shift Theorem
            'bins' can be fractional.  The resulting vector will have
            the same length as the original.
    """
    arr = np.asarray(arr)
    freqs = np.arange(arr.size / 2 + 1, dtype=float)
    phasor = np.exp(complex(0.0, 2 * np.pi) * freqs * bins / float(arr.size))
    return np.fft.irfft(phasor * np.fft.rfft(arr), arr.size)


def delay_from_DM(DM, freq):
    """Return the delay caused by DM at freq (MHz)"""
    return DM / (0.000241 * freq**2)

def onewrap_deltaDM(Pspin, Flo=400.0, Fhi=800.0):
    """Return the deltaDM where the dispersion smearing is one pulse period in duration"""
    return Pspin * 0.000241 / (1.0 / Flo**2 - 1.0 / Fhi**2)

def smear_impulse(Pspin, nDMs, nbins = default_nbins,freqs = default_freqs):
    # A pseudo-impulse response
    impulse = np.zeros(nbins)
    impulse[0] = 1.0

    deltaDM = onewrap_deltaDM(Pspin)
    DMs = np.linspace(0, 3 * deltaDM, nDMs)
    sums = np.zeros((len(DMs), nbins))
    i = 0

    #iterate over DMs and disperse the impulse function 
    for DM in DMs:
        sum = np.zeros(nbins)
        hidelay = delay_from_DM(DM, freqs[-1])
        for freq in freqs:
            #this converts bin delay in time to bin delay in phase
            bins = -(delay_from_DM(DM, freq) - hidelay) / Pspin * nbins
            #now implement delay
            sum += fft_rotate(impulse, bins)
        sums[i] = sum
        i += 1

    return sums, DMs, deltaDM

def get_kernels(Pspin, nDMs, nbins = default_nbins, freqs = default_freqs):

    sums, DMs, deltaDM = smear_impulse(Pspin, nbins, freqs)
    kernels = np.zeros(sums.shape).astype('complex_')
    kernel_scaling = DMs/deltaDM
    maxamp = max(fft(sums[0]))
    for i in range(nDMs):
        sum_fft = fft(sums[i])/maxamp
        kernels[i] = sum_fft

    return kernels, kernel_scaling

def sinc(x):
    return np.sin(x)/x

def gaussian(x, mu = 0.4, sig = 0.02):
    return np.exp(-0.5*((x - mu)/sig)**2)/(sig*np.sqrt(2*np.pi))

def disperse(prof, deltaDM, DM_labels, kernels, kernel_scaling):
    '''
    This function disperses an input pulse profile over a range of -2*deltaDM to 2*deltaDM according
    to the algorithm specified above.

    Inputs:
    -------
            prof (arr)     : 1D array of length 1024 specifying the signal at each phase bin
            deltaDM (float): the DM at which the pulse is smeared by one period
            DM_labels (arr): 1D array specifying the DM labels in the target power spectrum
            kernels (arr)  : 2D array containing the smeared impulse function kernels

    Returns:
    --------
            dispersed_prof_fft (arr): a 2D array of size (len(DM_labels), len(prof)) containing the
                                      dispersed profile values
    '''
    
    #take the fft of the pulse
    prof_fft = fft(prof)

    #i is our index referring to the DM_labels in the target power spectrum
    #find the starting index, where the DM scale is -2
    i = np.argmin(np.abs(-2*deltaDM - DM_labels))
    i0 = np.argmin(np.abs(DM_labels))
    #find the stopping index, where the DM scale is +2
    i_max = np.argmin(np.abs(2*deltaDM - DM_labels))

    dispersed_prof_fft = np.zeros((len(DM_labels), len(prof)), dtype = 'complex_')

    while i <= i_max:
        #j is the index referring to the closest value in DMs
        '''this can def be optimized by figuring out how many DM bins fit into
        each kernel bin between diff dms and then increasing j by that amount each iteration
        but i will save that for future optimization'''
        key = np.argmin(np.abs(np.abs(DM_labels[i]/deltaDM) - kernel_scaling))
        dispersed_prof_fft[i] = prof_fft * kernels[key]
        i += 1

    return dispersed_prof_fft, i0

def harmonics(prof_fft, f_true, df, n_harm):
    
    """This function calculates an array of harmonics for a given profile.

    Inputs:
    _______
            prof (ndarray): pulse phase profile
            f_true (float): true rotational frequency in Hz
            df (float)    : frequency bin width in target spectrum

    Returns:
    ________
            harmonics (ndarray) : Fourier-transformed harmonics of the profile convolved 
            with [cycles] number of Delta functions
    """
    #currently we are calculating the first 10 harmonics in the 2 bins on either side of...
    #the true value (2 + 2 = 4)
    harmonics = np.zeros((4*n_harm))
    bins = np.zeros((4*n_harm)).astype(int)
    
    #now evaluate sinc-modified power at each of the first 10 harmonics
    for i in range(1, n_harm + 1):
        f_harm = i*f_true
        bin_true = f_harm/df
        bin_below = np.floor(bin_true)
        bin_above = np.ceil(bin_true)

        #use 2 bins on either side
        current_bins = np.array([bin_below - 1, bin_below, bin_above, bin_above + 1])
        bins[(i - 1)*4:(i - 1)*4+4] = current_bins
        amplitude = prof_fft[i]*sinc(np.pi*(bin_true - current_bins))
        harmonics[(i - 1)*4:(i - 1)*4+4] = np.abs(amplitude)**2

    return bins, harmonics

def smear_harms(pulse, Pspin, sigma, df, nfreqbins, deltaDM, DM_labels, 
                kernels, kernel_scaling, noise = False, nu = 2):
    dispersed_prof_fft, i0 = disperse(pulse, deltaDM, DM_labels, kernels, kernel_scaling)

    f_nyquist = np.floor(df*nfreqbins / 2)
    f = 1/Pspin
    n_harm = int(np.floor(f_nyquist / f))
    print(f'n_harm = {n_harm}')

    bins_i0, harm_i0 = harmonics(dispersed_prof_fft[i0], 1/Pspin, df, n_harm)
    poweri0 = np.sum(harm_i0)

    if noise:
        #average power across bins will be equal to degrees of freedom
        #normalized by 1/2
        #such that for a 1-day stack (nu = 2) the average power is 1
        #for a 2-day stack (nu = 4) the average power is 2
        #etc...
        smeared_harm = stats.chi2.rvs(nu, size = (len(DM_labels), nfreqbins))/2
        power = sigma_to_power(sigma, nu) - nu/2

    else:
        smeared_harm = np.zeros((len(DM_labels), nfreqbins))
        #is this right? not a chi-squared distribution at all...
        power = sigma_to_power(sigma, nu)

    for i in range(len(smeared_harm)):
        bins, harm = harmonics(dispersed_prof_fft[i], 1/Pspin, df, n_harm)
        smeared_harm[i, bins] += harm*power/poweri0

    #smeared_i0 = np.copy(smeared_harm[i0])

    return smeared_harm, bins

def x_to_chi2(x, nu):
    
    #Note about A&S 26.4.16: I found this returned higher-error results than the 
    # x > 30 approximation
    
    if x <= 7.3:
        mean = 0.0
        sd = 1.0
        
        # Calculate the cumulative distribution function (CDF) of normal distribution
        p = stats.norm.cdf(x, loc=mean, scale=sd)
        
        # Calculate the cumulative distribution function (CDF) of chi-squared distribution
        chi2 = stats.chi2.ppf(p, nu)

        return chi2
    
    else:
        #A&S 26.2.23. This is the inverse of the operation for logp in PRESTO
        A = 2.515517
        B = 0.802853
        C = 0.010328
        D = 1.432788
        E = 0.189269
        F = 0.001308
        const = np.array([F, E - x*F, D - C - x*E, 1 - B - x*D, -(x + A)])
        t = np.roots(const)[1]
        prob = np.exp(-t**2/2)
        
        chi2 = chdtri(nu, prob)

        return chi2

def sigma_to_power(sigma, nu):

    chi2 = x_to_chi2(sigma, nu)
    power = chi2/2
    
    return power

def fold_harmonics(power_spectrum, bins, deltaDM, DM_labels, nu = 2,
                   harms_to_use = 'all'):
    
    i = 0
    DM_start = np.argmin(np.abs(-0.5*deltaDM - DM_labels))
    DM_stop = np.argmin(np.abs(0.5*deltaDM - DM_labels))
    stamp = np.zeros((len(power_spectrum[DM_start:DM_stop]), 104)) 
    #keep 50 bins on either side of harmonic

    #figure out how many harmonics we want to fold over
    if harms_to_use != 'all':
        bins = np.copy(bins[:4*harms_to_use])
        #note: the number of harms to add is GREATER than the actual number of 
        #harmonics before the Nyquist cutoff frequency, then the signal will
        #decrease, as we will be adding pure noise to the stamp
            
    while i < len(bins):
        freq_start = bins[i]
        if i == 0 and bins[i] < 50:
            temp_stamp = stats.chi2.rvs(nu, size = stamp.shape)/2
            temp_stamp[:, :freq_start+54] = power_spectrum[DM_start:DM_stop,
                                                             :freq_start+54]
            stamp += temp_stamp

        else:
            stamp += power_spectrum[DM_start:DM_stop, 
                                    freq_start - 50:freq_start+54]

        i += 4

    return stamp


