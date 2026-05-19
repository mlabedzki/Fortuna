import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from . import analytics as an
from . import utils as ut

def half_life(y):
    y_lag = y.shift(1).dropna()
    dy = y.diff().dropna()
    
    # Align the series for regression
    df = pd.concat([dy, y_lag], axis=1).dropna()
    df.columns = ['dy', 'y_lag']
    
    # Add a constant for the intercept
    #X = sm.add_constant(df['y_lag'])
    X = np.c_[np.ones(df.shape[0]),df['y_lag']]
    
    #model = sm.OLS(df['dy'], X)
    #res = model.fit()
          
    # The mean reversion coefficient (lambda) is the slope
    # Negative value is expected for mean-reverting series
    #lmbda = res.params['y_lag']
    lmbda = an.ols_coef(X,df['dy'])[1]
    
    # Calculate the half-life
    return float(-np.log(2) / lmbda)

def calculate_period_with_fft(timestamps, values):
    N = len(values)
    # Calculate the time step (assuming uniform sampling)
    T = timestamps[1] - timestamps[0]

    # Perform FFT
    yf = sp.fft.fft(values)
    xf = sp.fft.fftfreq(N, T)[:N//2] # Frequencies

    # Get the index of the peak frequency (excluding the DC component at index 0)
    dominant_frequency_idx = np.argmax(np.abs(yf[1:N//2])) + 1
    dominant_frequency = xf[dominant_frequency_idx]
    period = 1 / dominant_frequency

    print(f"Dominant Frequency (Hz): {dominant_frequency}")
    print(f"Estimated Period (time units): {period}")
    return period

def calculate_periodic_times_series(series_data: pd.Series, average=None):
    """
    Calculates the average period of a periodic time series with NaNs,
    after interpolating missing values. Returns the result in days.
    """
    # Create a copy and interpolate missing values to ensure smooth transitions
    # Choose an appropriate interpolation method (linear is good for sine waves)
    #interpolated_series = series_data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    #if interpolated_series.isna().any():
    #    print("Warning: Data still contains NaNs after interpolation and fill operations.")

    values = series_data.values
    timestamps = series_data.index.to_numpy()

    # Calculate the mean using nanmean, which is safe even if NaNs persist
    if average is None:
        mean_val = np.nanmean(values) 
    else:
        mean_val = average
    #print(f"Mean value of the series: {mean_val:.2f}")

    # Find indices where the value crosses the mean
    signs = np.sign(values - mean_val)
    # Identify where the sign changes. NaNs in signs will prevent correct comparison.
    sign_changes = (signs != np.roll(signs, 1))
    sign_changes[0] = False # Ignore the first point

    # Get the timestamps corresponding to the mean crossings
    mean_crossing_timestamps = timestamps[np.where(sign_changes)]

    if len(mean_crossing_timestamps) > 1:
        # Calculate time differences between consecutive mean crossings
        time_diffs = np.diff(mean_crossing_timestamps)

        # Convert the numpy timedelta64 array to days as a float
        # This handles cases where the index is datetime/timedelta type
        if time_diffs.dtype == np.dtype('<m8[ns]'): # Check if it is a timedelta type
            time_diffs_days = time_diffs / np.timedelta64(1, 'D')
        else:
            # Assume numeric index if not a timedelta
            time_diffs_days = time_diffs
        
        avg_half_period_days = np.nanmean(time_diffs_days) # Use nanmean just in case
        avg_period_days = avg_half_period_days * 2

        #print(f"Average time between mean crossings (Half Period): {avg_half_period_days:.2f} days")
        #print(f"Estimated Period: {avg_period_days:.2f} days")
        return avg_period_days
    else:
        #print("Not enough mean crossings found to estimate a period.")
        return np.nan

def butter_lowpass_filter(data, cutoff = 2, fs = 30, order = 2):
    data = data.ffill()
    nancount = np.isnan(data).sum()
    idx = data.index
    data = data.dropna()
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = sp.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = sp.signal.filtfilt(b, a, data)
    y = pd.Series(np.r_[np.array([np.nan] * nancount),y],index=idx)
    return y

def mixed_ema(y,period=252):
    shortmean = y.ewm(span=5).mean()
    longmean = y.ewm(span=252).mean()  
    #shortrel = y/shortmean - 1
    longrel = y/longmean - 1
    #longhl = longrel.rolling(period).apply(lambda x: min(252*2,max(1,ml.half_life(x))))
    #shorthl = shortrel.rolling(period).apply(lambda x: min(252*2,max(1,ml.half_life(x))))
    #shortper = shortrel.rolling(period).apply(lambda x: min(252*2,max(1,calculate_periodic_times_series(x))))
    longper = longrel.rolling(252).apply(lambda x: min(252*2,max(1,calculate_periodic_times_series(x))))    
    ratio = (252 - longper/2) / (252-5)
    return (1-ratio) * longmean + ratio*shortmean

def denoising(levels,i,ma="sma",begin=None,end=None,mod=0.2): 
    base = 1
    span = base*i #2/i-1
    if ma=="sma":
        ma_perf = np.log(levels).rolling(window=i*base).mean().shift(-int(i*base*mod))
    elif ma=="ema":
        ma_perf = np.log(levels).ewm(span=i*base).mean().shift(-int(i*base*mod))
        #ma_perf = np.log(levels).ewm(alpha=i).mean().shift(-int(span*mod))
    elif ma=="dema":
        ma_perf = an.DEMA(np.log(levels),i*base).shift(-int(i*base*mod))
        #ma_perf = an.DEMA(np.log(levels),i*base,i).shift(-int(span*mod))
    elif ma=="zlema":
        ma_perf = an.ZLEMA(np.log(levels),i*base).shift(-int(i*base*mod))    
        #ma_perf = an.ZLEMA(np.log(levels),i*base,i).shift(-int(span*mod))
    elif ma=="kama":
        ma_perf = an.KAMA(np.log(levels), n=i*base, fast_period=2, slow_period=i*base*2)
    volrel = float(ut.trimdf(an.emavol(ma_perf.diff()),begin,end).mean()/ut.trimdf(an.emavol_level(levels),begin,end).mean())
    mape = float(ut.trimdf(abs(np.log(levels) - ma_perf),begin,end).mean())
    avgperiods = calculate_periodic_times_series(np.log(levels)/ma_perf,1)
    return (i,volrel,mape,avgperiods,int(span*mod))

def wavegram(y, fs=1.0, plot=False, n=None):
    """
    Function to plot period length vs spectral density.
    
    Parameters:
    y (array_like): The time series data.
    fs (float): The sampling frequency of the time series (default is 1.0).
    """
    # Calculate the power spectral density using Welch's method
    # Frequencies are returned in units of Hz (cycles per unit time)
    freqs, spec = sp.signal.welch(y, fs=fs, scaling='density')
    #freqs, psd = sp.signal.periodogram(y, fs=1.0, window='boxcar', scaling='density')
    
    # Calculate period length (1/frequency). 
    # Frequencies near zero are excluded to avoid division by zero or infinite period lengths.
    non_zero_freqs = freqs[freqs > 0]
    non_zero_spec = spec[freqs > 0]
    period_length = 1 / non_zero_freqs

    if n is None:
        # n defaults to the max period length (min frequency)
        n = period_length[0]
        
    if plot is not False:
        # Plot period length vs spectral density
        plt.figure(figsize=(10, 6))
        # Removed the 'type' argument and use 'linestyle' or just the default behavior
        plt.plot(period_length, non_zero_spec, linestyle='-') 
        plt.xlim(0, np.log(n))
        plt.xlabel('Period Length')
        plt.ylabel('Spectral Density')
        plt.title('Falogram (Periodogram)')
        plt.grid(True)
        plt.show()

    out = np.vstack((period_length, spec)).T
    # Subset the output dataframe/array
    subset_out = out[out[:, 0] <= n]
        
    return subset_out

def p9010(x):
    """Calculates the difference between the 90th and 10th percentiles."""
    return float(np.percentile(x, 90) - np.percentile(x, 10))

def wave_ratio(x):
    amp = p9010(x)
    ff = wavegram(x, plot=False)
    m = ff[np.argmax(ff[1:, 1]) + 1, 0] # Finds the frequency (col 0) at the peak amplitude (col 1)
    avg = np.sum(ff[:, 0] * ff[:, 1]) / np.sum(ff[:, 1])
    disty = x[-1] / (amp / 2)
    # This part requires a specific zero-crossing detection implementation, simplified here:
    diff_sign = np.diff(np.sign(np.diff(x)))
    # Indices for local minima/maxima (zero crossings of second derivative sign change)
    maxima_indices = np.where(diff_sign < 0)[0] + 2 # Adjusting for diff operations offset
    minima_indices = np.where(diff_sign > 0)[0] + 2
    distmax = (len(x) - maxima_indices[-1]) / m if maxima_indices.size > 0 else np.nan
    distmin = (len(x) - minima_indices[-1]) / m if minima_indices.size > 0 else np.nan  
    return np.array([amp, avg, m, disty, distmax, distmin])

class SSA(object):
    
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    
    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)

def low_pass_filter_fourier(ts_data, num_harmonics):
    """
    Filters a time series using a specific number of Fourier harmonics.
    """
    N = len(ts_data)
    fourier_coeffs = np.fft.fft(ts_data)
    
    # Create a truncated version of the coefficients
    # Coeffs are symmetric, we need to keep the first N and the last N
    truncated_coeffs = np.zeros(N, dtype=complex)
    
    # Keep the fundamental (DC component) and the first N harmonics
    truncated_coeffs[0:num_harmonics + 1] = fourier_coeffs[0:num_harmonics + 1]
    # Keep the corresponding negative frequency components (for real-valued output)
    truncated_coeffs[N - num_harmonics:N] = fourier_coeffs[N - num_harmonics:N]
    
    # Inverse FFT to reconstruct the signal
    fitted_signal = np.fft.ifft(truncated_coeffs)
    
    return np.real(fitted_signal)

def forecast_fourier(ts_data, num_harmonics, num_forecast_elements):
    """
    Forecasts a time series using a truncated Fourier series model.
    Assumes ts_data covers one full period.
    """
    N_original = len(ts_data)
    fourier_coeffs = np.fft.fft(ts_data)

    # Truncate high-frequency coefficients to remove noise
    truncated_coeffs = np.zeros(N_original, dtype=complex)
    truncated_coeffs[0:num_harmonics + 1] = fourier_coeffs[0:num_harmonics + 1]
    # Include symmetric negative frequency components
    truncated_coeffs[N_original - num_harmonics:N_original] = fourier_coeffs[N_original - num_harmonics:N_original]

    # Function to reconstruct the signal at arbitrary time indices
    def reconstruct_ts_scaled(n_indices, coeffs, N_len):
        reconstructed_signal = np.zeros(len(n_indices), dtype=complex)
        for k in range(N_len):
            if np.abs(coeffs[k]) > 1e-9:
                # The core formula: sum(Ck * exp(i * 2 * pi * k * n / N))
                angular_freq = 2 * np.pi * k / N_len
                term = coeffs[k] * np.exp(1j * angular_freq * n_indices)
                reconstructed_signal += term
        # Apply the 1/N scaling once
        return np.real(reconstructed_signal) * (1/N_len)

    # Time indices for the forecast period (e.g., indices 200 to 229)
    forecast_indices = np.arange(N_original, N_original + num_forecast_elements)
    
    # Calculate the forecast
    forecasted_elements = reconstruct_ts_scaled(forecast_indices, truncated_coeffs, N_original)

    return forecasted_elements

def analyze_fft_magnitudes(ts,mod=1):
    # To help determine N:
    fourier_coeffs = np.fft.fft(ts)
    # We only need the first half (up to the Nyquist frequency)
    magnitudes = np.abs(fourier_coeffs[:len(ts) // 2])
    # Optional: normalize or sort magnitudes to find a threshold
    
    plt.figure(figsize=(8, 4))
    plt.plot(np.log(magnitudes[:int(len(magnitudes)*mod)]))
    plt.title('Power Spectrum Magnitude')
    plt.xlabel('Harmonic Index (Frequency)')
    plt.ylabel('Log of Magnitude')
    plt.grid(True, linestyle='dashed')
    plt.show()
    
# import pyrssa as prs #use this or SSA class as alternative
# # uncomment below if you plan to use pyrssa
    
def wave_anal(y):
    import pyrssa as prs
    # tested for series of 2500 elements
    s = prs.SSA(y, neig=15) # Assuming window length is default or determined automatically
    # Reconstruct the series into groups
    # Groups are 1:3, 4:5, 6:15
    r = prs.reconstruct(s,[slice(0, 3), slice(3, 5), slice(5, 15)])
    tt = y - r.F1 - r.F2 - r.F3
    s2 = prs.SSA(tt, neig=15)
    rr = prs.reconstruct(s2,[slice(0, 2), slice(2, 4), slice(4, 10)]) # Groups 1:2, 3:4, 5:10
    # Calculate wave ratios
    ratio_r3 = wave_ratio(r.F3)
    ratio_rr1 = wave_ratio(rr.F1)
    ratio_rr2 = wave_ratio(rr.F2)
    ratio_rr3 = wave_ratio(rr.F3)
    ratio_rr123 = wave_ratio(rr.F1 + rr.F2 + rr.F3)
    ratio_tt = wave_ratio(tt)
    return np.array([ratio_r3, ratio_rr1, ratio_rr2, ratio_rr3, ratio_rr123, ratio_tt])

def component_wnorm(x, neig=10):
    """
    Calculates normalized component weights.
    
    Args:
      x: An object with methods nsigma(), wnorm(), and attribute sigma (array-like).
      neig: The number of neighbors/components to consider (default 10).
    
    Returns:
      A list of the normalized, rounded component weights as percentages.
    """
    import pyrssa as prs
    idx_len = min(x.nsigma(), neig)
    idx = range(idx_len)
    total = prs.wnorm(x)**2
    # Select relevant sigma values and perform calculation
    weights = [100 * (x.sigma[i]**2) / total for i in idx]
    return [round(w, 2) for w in weights]

def corr_fcst_lvl(y,n):
    out = []
    for x in y:
        #SSA forecast level starting from last fitted value, we move all fcst up or down to match last realised value level
        dd = x.iloc[n]-x.iloc[n-1]
        #dd = dd - (x.iloc[n-1]-x.iloc[n-2]) # correct for natural trend to make it other than 0
        x = x - dd
        x.iloc[0:n] = x.iloc[0:n] + dd
        out.append(x)
    return out
    
def ssa_backtest(y,neig,span,start=2500,epochs=5570):
    import pyrssa as prs
    mat = np.zeros((epochs,5))
    mat_2 = np.zeros((epochs,5))
    delay = int(0.2*span)
    ema_imperf = an.EMA(np.log(y),span).shift(-0) #perf version has delay instead of 0
    for i in range(0,epochs):
        signal = np.log(y).iloc[0:(start+i)] 
        s = prs.ssa(signal, neig = neig)
        rf = corr_fcst_lvl([prs.rforecast(s, groups=[range(1, neig+1)], length = delay*3, only_new=False, base = "reconstructed", reverse = False, drop = True, drop_attributes = False, cache = True)],len(signal))
        mat[i,0] = rf[0].iloc[start+i-1-delay]
        mat[i,1] = rf[0].iloc[start+i-1]
        mat[i,2] = rf[0].iloc[start+i-1+delay]
        mat[i,3] = rf[0].iloc[start+i-1+delay*2]
        mat[i,4] = rf[0].iloc[start+i-1+delay*3]
        signal2 = ema_imperf.iloc[0:(start+i)] 
        s2 = prs.ssa(signal2, neig = neig)
        rf2 = corr_fcst_lvl([prs.rforecast(s2, groups=[range(1, neig+1)], length = delay*3, only_new=False, base = "reconstructed", reverse = False, drop = True, drop_attributes = False, cache = True)],len(signal))
        mat_2[i,0] = rf2[0].iloc[start+i-1-delay]
        mat_2[i,1] = rf2[0].iloc[start+i-1]
        mat_2[i,2] = rf2[0].iloc[start+i-1+delay]
        mat_2[i,3] = rf2[0].iloc[start+i-1+delay*2]
        mat_2[i,4] = rf2[0].iloc[start+i-1+delay*3]
    mat = pd.DataFrame(mat,index=y.index[(start-1):(epochs-1)])    
    mat_2 = pd.DataFrame(mat_2,index=y.index[(start-1):(epochs-1)])
    return (mat,mat_2)