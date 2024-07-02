import numpy as np
from scipy.fft import fft
from scipy.stats import chi2

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

    return dispersed_prof_fft

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

def smear_harms(pulse, Pspin, sigma, df, n_harm, nfreqbins, deltaDM, DM_labels, 
                kernels, kernel_scaling, noise = False, nu = 2):
    dispersed_prof_fft = disperse(pulse, deltaDM, DM_labels, kernels, kernel_scaling)

    if noise:
        #average power across bins will be equal to degrees of freedom
        #normalized by 1/2
        #such that for a 1-day stack (nu = 2) the average power is 1
        #for a 2-day stack (nu = 4) the average power is 2
        #etc...
        smeared_harm = chi2.rvs(nu, size = (len(DM_labels), nfreqbins))/2

    else:
        smeared_harm = np.zeros((len(DM_labels), nfreqbins))

    for i in range(len(smeared_harm)):
        bins, harm = harmonics(dispersed_prof_fft[i], 1/Pspin, df, n_harm)
        smeared_harm[i, bins] = harm

    return smeared_harm