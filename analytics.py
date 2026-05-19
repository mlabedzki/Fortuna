import pandas as pd
import datetime as dt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from . import strategies as st
import tsfresh as tsf #for some TA indicators
#import ta #if you want to add more TA indicators

def maxdd_abs(eq,stress=None):
    dd = eq.div(eq.cummax()).sub(1)
    if stress is not None:
        dd += stress
    mdd = float(dd.min())
    end = int(dd.argmin())
    start = int(eq.iloc[:end].argmax())
    return mdd, start, end
    
def maxdd_rel(returns,stress=None):
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    if stress is not None:
        dd += stress    
    mdd = float(dd.min())
    end = int(dd.argmin())
    start = int(r.iloc[:end].argmax())
    return mdd, start, end

def SR(x,b=252,con=False):
    if isinstance(x, st.Portfolio):
        # Access internal attributes
        y = x.calculate_portfolio_returns()       
    else:
        #y = np.log(1+x)
        y=x
    if con:
        nas_pct = 0
    else:    
        nas_pct = y.isna().sum() / len(y)
    return float(np.round(np.nanmean(y)/(np.nanstd(y)) * np.sqrt(b * (1-nas_pct)),4))
    
def calculate_rolling_sharpe(returns_series, window=252, benchmark=None, b=252, rate=0, overlap=1):
    """
    Takes a series of daily returns and returns a rolling 1-year 
    annualized Sharpe ratio series.
    """
    # 1. Annualization factor for daily returns
    # annual_factor = 1#np.sqrt(252)
    
    # 2. Define custom function for the window
    # Assumes excess returns (returns - risk_free_rate)
    def rolling_sharpe(window_returns, benchmark=benchmark, b=b, rate=rate, overlap=overlap):
        #return (window_returns.mean() / window_returns.std()**2) * annual_factor
        #return window_returns.skew()
        return SR(window_returns)
        #g_dict = strat_stats(window_returns)
        #return np.array(list(g_dict.items()))
    
    # 3. Apply the rolling calculation
    rolling_sharpe_series = returns_series.rolling(window=window).apply(rolling_sharpe)
    
    return rolling_sharpe_series

def strat_stats(x, benchmark=None, b=252, rate=0, overlap=1, window=None):
    """
    Assumes x is a pandas Series or DataFrame of returns.
    """
    # Handle multivariate input (column-wise mean)
    if isinstance(x, pd.DataFrame) and x.shape[1] > 1:
        # Note: In R, this calls custom log/std conversions; 
        # using simple mean of returns as the standard equivalent.
        x = x.mean(axis=1)
    if isinstance(x, st.Portfolio):
        # Access internal attributes
        x = x.calculate_portfolio_returns()   
        
    # Clean data: na20 (replace NAs with 0) and removing NAs
    x_no_na = x.dropna()
    x_filled = x.fillna(0)

    lm0 = OLSprepnfit(x_filled.cumsum(),pd.Series(np.arange(len(x_filled)),index=x_filled.index))
    if window is not None:
        rolling_sharpe_series = calculate_rolling_sharpe(x_filled, window=window, benchmark=None, b=b, rate=rate, overlap=overlap).dropna()
        
    else:
        rolling_sharpe_series = [0,0]
        
    if benchmark is not None:
        lm1 = OLSprepnfit(x_filled*b,benchmark*b)
        alpha = lm1.params.iloc[0]
        beta = lm1.params.iloc[1]
        alpha_b1 = lm1.params.iloc[0]/lm1.params.iloc[1]
    else:
        alpha = np.nan
        beta = np.nan
        alpha_b1 = np.nan

    # Metrics
    nas_pct = x.isna().sum() / len(x)
    skew = x_no_na.skew()
    
    # Calculate Averages (Annualized)
    avg0 = (x_filled * b - rate).mean()
    avg = (x_no_na * b - rate).mean()
    # avg_scaled = (x_no_na * b * (1-nas_pct) - rate).mean()
    
    # Standard Deviations (Annualized)
    sd0 = x_filled.std() * np.sqrt(b)
    sd = x_no_na.std() * np.sqrt(b)
    # sd_scaled = x_no_na.std() * np.sqrt(b * (1-nas_pct))
    
    # Semi-Deviation (for Sortino)
    negative_returns = x_filled[x_filled < 0]
    sdneg0 = np.sqrt((negative_returns**2).mean()) * np.sqrt(b)
       
    # Profit Factor
    pos_sum = x_no_na[x_no_na >= 0].sum()
    neg_sum = abs(x_no_na[x_no_na < 0].sum())
    pf = pos_sum / neg_sum if neg_sum != 0 else np.nan
        
    # Max Drawdown (Simple cumulative wealth peak-to-trough)
    cum_ret = (1 + x_filled).cumprod()
    max_dd = (cum_ret.div(cum_ret.cummax()) - 1).min()
    
    # Ulcer Index, aka Integral over DD's
    max_x = x_filled.cummax()
    drawdown = 100 * (x_filled - max_x) / max_x
    UI = np.sqrt((drawdown**2).mean())
    
    # # 6. Rolling Max Drawdown (with Overlap)
    # # We use a rolling window to find the peak within the 'overlap' period
    # wealth_index = (1 + x_no_na).cumprod()
    # # Peak value within the rolling window
    # rolling_peak = wealth_index.rolling(window=overlap, min_periods=1).max()   
    # # Current drawdown relative to that rolling peak
    # drawdowns = (wealth_index / rolling_peak) - 1.0
    # # Max drawdown is the minimum (most negative) value in that series
    # max_dd = abs(drawdowns.min())

    # Compile Results
    out = {
        'SR': float(avg0 / sd0) if sd0 != 0 else 0,
        'minrollSR': min(rolling_sharpe_series),
        'stdrollSR': float(np.std(rolling_sharpe_series)),
        'Sortino': float(avg0 / sdneg0) if sdneg0 != 0 else 0,
        'Calmar': float(avg0 / abs(max_dd)) if max_dd != 0 else 0,
        'Linearity': float(lm0.rsquared_adj),
        # 'LinRegResid': float(lm0.mse_resid),
        'Skew': float(skew),
        'Return': float(avg0), # Simplified representation
        'Vol': float(sd0), # Simplified representation
        'MaxDD': abs(max_dd),
        'UlcerI': float(UI), 
        'NAs%': float(nas_pct),
        'PF': float(pf),
        #'SR.ACT': float(avg_scaled / sd_scaled) if sd_scaled != 0 else 0, #in fact it should be approx. avg0/sd0
        'SR.CON': float(avg / sd) if sd != 0 else 0,
        'alpha': float(alpha),
        'beta': float(beta),
        'alpha_b1': float(alpha_b1)       
    }
    return {k: round(v, 3) for k, v in out.items()}
 
def omega_ratio(returns, threshold=0):
    """
    Calculates the Omega Ratio.
    
    :param returns: Array of periodic returns (decimal format, e.g., 0.01 for 1%).
    :param threshold: The Minimum Acceptable Return (MAR). Defaults to 0.
    """
    returns = np.array(returns)
    
    # Calculate returns relative to the threshold
    diff = returns - threshold
    
    # Sum of positive differences (gains)
    gains = np.sum(diff[diff > 0])
    
    # Sum of absolute negative differences (losses)
    losses = np.sum(np.abs(diff[diff < 0]))
    
    if losses == 0:
        return np.inf
        
    return gains / losses

def emavol(returns,periods=63,pinyear=252):
    return np.sqrt((returns**2).ewm(span=periods).mean()*pinyear)

def smavol(returns,periods=63,pinyear=252):
    return np.sqrt((SMA(returns**2,periods) - SMA(returns,periods)**2)*pinyear)

def emavol_level(levels,periods=63,pinyear=252):
    returns = np.log(levels).diff()
    return np.sqrt((returns**2).ewm(span=periods).mean()*pinyear)

def lagmonth(df,months=1):
    return pd.DataFrame(df.values, index = df.index + dt.timedelta(days=1+31*months), columns=df.columns).resample('ME').last()
   
def is_invertible(A):
    out = A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]
    return out

def ols_coef(x,y):
    xxinv=np.dot(x.T,x)
    if is_invertible(xxinv):
        xxinv = np.linalg.inv(xxinv)
        xy=np.dot(x.T,y)
        coef=np.dot(xxinv,xy)
    return coef

#vole^2 = vols^2 - b^2*volm^2
#w_0 = alpha*252 / vole^2 / (meanm / volm^2)
#w_A = w_0 / (1 + w_0 * (1-beta))

#L_strat = alpha / meanm * volm^2 / vole^2 = alpha / meanm * volm^2 / (vols^2 - b^2*volm^2)
#L_strat = alpha / (vols^2 - b^2*volm^2)

def unEMA(fcst,start,span):
    # ema[i] = alpha*y[i] + (1-alpha)*ema[i-1]
    # ema[i] = (alpha*fcst[i]+ (1-alpha)*ema[i-1] )/(1-alpha)
    alpha = 2./(span+1)
    ema = np.zeros(len(fcst))
    for i in range(0,len(fcst)):
        if i==0:
            ema[i] = alpha/(1-alpha) * fcst[i] + start
        else:
            ema[i] = alpha/(1-alpha) * fcst[i] + ema[i-1]
    return ema

def SMA(y,span):
    ema = y.ewm(span=span,adjust=False,ignore_na=True).mean().iloc[0:span-1]
    sma = y.rolling(window=span).mean()
    ratio = sma.iloc[span-1]/ema.iloc[span-2]
    sma.iloc[0:span-1] = ema*ratio
    return sma

def EMA(y,span,alpha=None):
    if alpha is None:
        ema = y.ewm(span=span,adjust=False,ignore_na=True).mean()
    else:
        ema = y.ewm(alpha=alpha,adjust=False,ignore_na=True).mean()
    return ema

def DEMA(y,span,alpha=None):
    if alpha is None:
        ema = y.ewm(span=span,adjust=False,ignore_na=True).mean()
        emaema = ema.ewm(span=span,adjust=False,ignore_na=True).mean()
    else:
        ema = y.ewm(alpha=alpha,adjust=False,ignore_na=True).mean()
        emaema = ema.ewm(alpha=alpha,adjust=False,ignore_na=True).mean()        
    return 2*ema - emaema

def ZLEMA(y,n,alpha=None):
    if alpha is not None:
        n = 2/alpha-1
        #alpha = 2/(n+1)
    lag = int((n-1)/2)
    new = y + (y-y.shift(lag))
    #it is the same as ema+adj, where ema = y.ewm(span=n).mean(); adj = (ema-ema.shift(lag))
    if alpha is None:
        ema = new.ewm(span=n).mean()
    else:
        ema = new.ewm(alpha=alpha).mean()
    return ema

def KAMA(price_series, n=10, fast_period=2, slow_period=30):
    """
    Calculates the Kaufman's Adaptive Moving Average (KAMA) using pandas and numpy.
    
    Args:
        price_series (pd.Series): The input price series (e.g., 'Close' prices).
        n (int): The number of periods for the Efficiency Ratio (ER) calculation (default 10).
        fast_period (int): The fast EMA period for the smoothing constant (default 2).
        slow_period (int): The slow EMA period for the smoothing constant (default 30).
        
    Returns:
        pd.Series: A Series containing the KAMA values.
    """
    price_series = price_series.ffill()
    nancount = np.isnan(price_series).sum()
    idx = price_series.index
    price_series = price_series.dropna()
    
    # 1. Efficiency Ratio (ER) calculation
    # Change = Abs(price – price(n periods ago))
    change = abs(price_series - price_series.shift(n))
    # Volatility = Sum of Abs(price – previous price) over n periods
    volatility = abs(price_series - price_series.shift(1)).rolling(window=n).sum()
    # ER = Change / Volatility
    er = change / volatility
    
    # Handle division by zero or NaN values in ER calculation
    er = er.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 2. Smoothing Constant (SC) calculation
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    # SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # 3. KAMA calculation (iterative process)
    kama = pd.Series(index=price_series.index)
    
    # Initialize the first KAMA value with the first valid price after the lookback period
    first_valid_index = price_series.index[n]
    kama[first_valid_index] = price_series[first_valid_index]
    
    # Iterate through the series to calculate KAMA for subsequent periods
    for i in range(n + 1, len(price_series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (price_series.iloc[i] - kama.iloc[i-1])
        
    return pd.Series(np.r_[np.array([np.nan] * nancount),kama],index=idx)

def LowPass(Data_Series, Period): #gives same output as DEMA
    """
    Implements a specific IIR low-pass filter defined by a custom difference equation,
    optimized to work with pandas Series input and output.
    
    Args:
        Data_Series (pd.Series): The input time series data as a pandas Series.
        Period (int or float): The period parameter used to calculate the smoothing factor.
        
    Returns:
        pd.Series: The filtered time series data (LP) with the original index preserved.
    """
    Data_Series = Data_Series.ffill()
    nancount = np.isnan(Data_Series).sum()
    idx = Data_Series.index
    Data_Series = Data_Series.dropna()
    
    # Ensure input is treated as numeric
    Data = Data_Series.astype(float)
    len_data = len(Data)
    
    # Initialize the output Series with NaNs, preserving the original index
    LP = pd.Series(index=Data.index, dtype=float)
    
    # Set initial conditions for the first two elements
    # Use .iloc for positional indexing
    if len_data > 0:
        LP.iloc[0] = Data.iloc[0]
    if len_data > 1:
        LP.iloc[1] = Data.iloc[1]
        
    # Calculate smoothing factors
    a = 2 / (Period + 1)
    a2 = a * a
    
    # Define coefficients based on the R formula
    coeff_data_i = (a - a2 / 4)
    coeff_data_i_minus_1 = 0.5 * a2
    coeff_data_i_minus_2 = -(a - 0.75 * a2)
    coeff_LP_i_minus_1 = 2 * (1 - a)
    coeff_LP_i_minus_2 = -(1 - a) * (1 - a)
    
    # Iterate from the third element (index 2 in Python positional indexing)
    for i in range(2, len_data):
        # Use .iloc for consistent positional access to both Data and LP Series
        LP.iloc[i] = (coeff_data_i * Data.iloc[i] + 
                      coeff_data_i_minus_1 * Data.iloc[i-1] + 
                      coeff_data_i_minus_2 * Data.iloc[i-2] + 
                      coeff_LP_i_minus_1 * LP.iloc[i-1] + 
                      coeff_LP_i_minus_2 * LP.iloc[i-2])
                 
    return pd.Series(np.r_[np.array([np.nan] * nancount),LP],index=idx)

def lsma(data, period=14, regression=True):
    size = len(data)
    out = np.full(size, np.nan)
    w = np.arange(1, period + 1, dtype=np.float64)
    if regression:
        for i in range(period - 1, size):
            e = i + 1
            s = e - period
            intercept, slope = np.dot(np.linalg.pinv(np.vstack((np.ones(period), w)).T), data[s:e])
            out[i] = slope * period + intercept
    else:
        for i in range(period - 1, size):
            e = i + 1
            s = e - period
            out[i] = np.dot(data[s:e], w) / np.sum(w)
    return out

def zlsma(data, period=14, regression=True):
    size = len(data)
    sum_w = np.sum(np.arange(1, period + 1, dtype=np.float64))
    lsma_v = lsma(data, period, regression)
    out = np.full(size, np.nan)
    w = sum_w / (2 * np.sum(np.arange(1, period)))
    for i in range(period - 1, size):
        out[i] = lsma_v[i] + (data[i] - lsma_v[i]) * w
    return out

def get_column_names(obj):
    if isinstance(obj, pd.DataFrame):
        # Returns all column names as a list
        return obj.columns.tolist()
    elif isinstance(obj, pd.Series):
        # Returns the Series name in a list (to keep output format consistent)
        return [obj.name]
    return []
    
def OLSprepnfit(y,x,const=True):
    df = pd.concat([y,x], axis=1)
    df.dropna(inplace=True)
    df.columns = ["y"]+get_column_names(x)
    if const:
        df = sm.add_constant(df)
    return sm.OLS(df["y"],df.loc[:,df.columns != 'y']).fit()

def get_weights(d, size):
    # Returns weights for the fractional differencing filter
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff(series, d, threshold=0.01, lin=False):
    # 1. Determine weights that are significantly above threshold
    # Higher threshold = shorter memory window (faster)
    w = get_weights(d, len(series))
    w_sum = np.cumsum(abs(w))
    w_sum /= w_sum[-1]
    skip = w_sum[w_sum > threshold].shape[0]
    w = w[-skip:]
    
    # 2. Apply weights to the series (Log Close is best input)
    res = {}
    if not lin:
        series_pd = np.log(series).ffill().dropna()
    else:
        series_pd = series.ffill().dropna()
    
    for i in range(skip, series_pd.shape[0]):
        res[series_pd.index[i]] = np.dot(w.T, series_pd.iloc[i-skip:i].values)[0]
        
    return pd.Series(res)
   
def CMO(close,window=252):
        """get Chande Momentum Oscillator

        The Chande Momentum Oscillator (CMO) is a technical momentum
        indicator developed by Tushar Chande.
        https://www.investopedia.com/terms/c/chandemomentumoscillator.asp

        CMO = 100 * ((sH - sL) / (sH + sL))

        where:
        * sH=the sum of higher closes over N periods
        * sL=the sum of lower closes of N periods
        """
        close_diff = close.diff()        
        up = close_diff.clip(lower=0)
        down = close_diff.clip(upper=0).abs()
        sum_up = up.rolling(window, min_periods=1, center=False).sum()
        sum_down = down.rolling(window, min_periods=1, center=False).sum()
        dividend = (sum_up - sum_down).values
        divisor = (sum_up + sum_down).values
        cmo = np.divide(100 * dividend, divisor,
                        out=np.zeros_like(dividend), where=divisor != 0)
        res = pd.Series(cmo, index=close.index)
        res.iloc[0] = 0.0
        return res    
    
def get_ohlc_features(df):
    # Ensure standard OHLCV column names
    h, l, c, o, v = df['High'], df['Low'], df['Close'], df['Open'], df['Volume']
    
    # 1. Garman-Klass Volatility
    # Uses OHLC to estimate volatility more efficiently than close-to-close
    raw_gk = 0.5 * np.log(h/l)**2 - (2*np.log(2)-1) * np.log(c/o)**2
    df['1d_GK_Vol'] = np.sqrt(raw_gk)
    
    # 1b. Compute 20-day rolling mean and take square root for volatility   
    df['GK_Vol'] = np.sqrt(raw_gk.rolling(window=20).mean())
    # df['GK_emavol'] = emavol(df['1d_GK_Vol'],21,1)
        
    # 2. ATR (Average True Range) - 14 period
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # 3. Bollinger Band Width - 20 period
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['BB_Width'] = (4 * std20) / ma20
    
    # 4. OBV (On-Balance Volume)
    df['OBV'] = (np.sign(c.diff()) * v).fillna(0).cumsum()
    
    # 5. CMF (Chaikin Money Flow) - 20 period
    mfv = (((c - l) - (h - c)) / (h - l)) * v
    df['CMF'] = mfv.rolling(20).sum() / v.rolling(20).sum()
    
    # 6. RSI (Relative Strength Index) - 14 period
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # 7. VWAP (Rolling 20-period)
    #df['VWAP'] = (v * (h + l + c) / 3).cumsum() / v.cumsum()   
    window = 20
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    pv = tp * df['Volume']
    df['VWAP_20'] = pv.rolling(window).sum() / df['Volume'].rolling(window).sum()
    
    # 8. Ulcer Index - 14 period
    # Measures stress by looking at the square root of mean squared drawdowns
    max_c = c.rolling(14).max()
    drawdown = 100 * (c - max_c) / max_c
    df['Ulcer_Index'] = np.sqrt((drawdown**2).rolling(14).mean())
    
    # 9. Z-Score (Rolling 20-day)
    df['Z_Score'] = (c - ma20) / std20
    
    # 10. Fourier Transform (Dominant Frequency Magnitude)
    # Extract the magnitude of the dominant cycle from the PREVIOUS 30 days
    def get_dominant_mag(x):
        fourier = np.abs(np.fft.fft(x)) if len(x) >= 30 else np.nan
        return np.max(fourier[1:]) # Ignore the DC component (index 0)
    def get_energy_ratio(x,num_segments=10, segment_focus=9):
        return tsf.feature_extraction.feature_calculators.energy_ratio_by_chunks(x, [{"num_segments": num_segments, "segment_focus": segment_focus}])[0][1]
        
    df['Fourier_Signal'] = c.rolling(30).apply(get_dominant_mag)  
    
    df['FracDiff_05'] = frac_diff(c, d=0.5)

    df['VR'] = np.log(df.Volume).diff() # Volume return
    df['VA'] = np.log(df.Volume/SMA(df.Volume,252)) # Volume vs Average
    df['HL'] = np.log(df.High/df.Low) # High-Low range
    df['BR'] = (df.Close - df.Open)/(df.High - df.Low) # Body range
    df['KAMArel'] = np.log(df.Close/KAMA(df.Close)) #10,2,30

    df['logret'] = np.log(df.Close).diff()
    # df['logreto'] = np.log(df.Open).diff()
    df['gap'] = np.log(df.Open/df.Close.shift(1)) # Opening Gap
    df['hv21'] = emavol(df['logret'],21,1)
    # df['hv21o'] = emavol(df['logreto'],21,1)
    # df['hv21gap'] = smavol(df['gap'],21,1)
    # df['hv21gap2'] = emavol(df['gap'],21,1)
    df['hv252'] = emavol(df['logret'],252,1)
    df['nlogret'] = (df['logret'])/df['hv21']
    df['ngap'] = (df['gap'])/df['hv21']
    df['volratio'] = np.log(df['hv21']/df['hv252'])
    df['dhv21'] = df['hv21'].diff(1)
    df['dlhv21'] = np.log(df['hv21']).diff(1)
    df['dnlogret'] = df['nlogret'].diff(1)
    df['anlogret'] = np.abs(df['nlogret'])
    df['adnlogret'] = np.abs(df['dnlogret'])
    df['adhv21'] = np.abs(df['dhv21'])
    df['adlhv21'] = np.abs(df['dlhv21'])
    
    # df['Energy_ratio_by_chunks__num_segments_10__segment_focus_9'] = c.rolling(252).apply(get_energy_ratio)
    df['ROC_1M'] = np.log(df.Close).diff(1*21)
    df['ROC_3M'] = np.log(df.Close).diff(3*21)
    df['ROC_6M'] = np.log(df.Close).diff(6*21)
    df['ROC_12M'] = np.log(df.Close).diff(12*21)
    # df['CMO'] = CMO(df.Close)
    # df['Time_reversal_asymmetry_statistic__lag_1'] = tsf.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(df.Close, lag=1)
    # df['Time_reversal_asymmetry_statistic__lag_2'] = tsf.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(df.Close, lag=2)
    # df['Time_reversal_asymmetry_statistic__lag_3'] = tsf.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(df.Close, lag=3)
    # df['Force_Inx'] = ta.volume.force_index(df.Close, df.Volume, window=13, fillna=False)
    # df['Plus_DI'] = ta.trend.adx_pos(h, l, c, window=14, fillna=False)
    # df['Minus_DI'] = ta.trend.adx_neg(h, l, c, window=14, fillna=False)
    
    return df

def get_ohlc_features_aux(df_og):
    df = df_og.copy()
    # Ensure standard OHLCV column names
    h, l, c, v = df['High'], df['Low'], df['Close'], df['Volume']

    window = 20
    tp = (h+l+c) / 3
    pv = tp * v
    VWAP_20 = pv.rolling(window).sum() / v.rolling(window).sum()
    
    df['ZLEMA'] = ZLEMA(c,30)
    df['ZLEMArel'] = np.log(c/df['ZLEMA'])
    df['KAMA'] = KAMA(c)
    df['longKAMA'] = KAMA(c,21*10,2,40)
    df['longZLEMA'] = ZLEMA(c,21*2)
    df['relVWAP_20'] = np.log(c/VWAP_20)

    logret = np.log(df.Close).diff()
    df['logretd'] = np.log(df.Close/df.Open)
    df['hv21day'] = emavol(df['logretd'],21,1)
    df['nlogretd'] = (df['logretd'])/df['hv21day']
    
    # 1c. logATR - something similar to GK
    high_low = np.log(h/l)
    high_prev_close = abs(np.log(h / c.shift(1)))
    low_prev_close = abs(np.log(l / c.shift(1)))
    # True Range is the maximum of the three
    df['ltr'] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    # ATR calculation using Wilder's Smoothing (Exponential Moving Average logic)
    # The first value is a simple mean, followed by the smoothing formula
    df['altr'] = df['ltr'].ewm(span=21, adjust=False).mean()
    df['nlogret2'] = (logret)/df['altr']
    target_vol=0.2
    df['dconvol'] = st.levret(target_vol/np.sqrt(252)/df['altr'].shift(1),logret)
    # df['dconvolday'] = st.levret(target_vol/np.sqrt(252)/df['altr'].shift(1),df['logretd'])
    # df['dconvolgap'] = st.levret(target_vol/np.sqrt(252)/df['altr'].shift(1),df['gap'])
    
    return df    

def weighted_ts_momentum(price):
    """
    Calculates weighted momentum for a daily price series:
    Momentum = (12*1m_ret + 4*3m_ret + 2*6m_ret + 12m_ret) / 4
    """
    # Define approximate trading days per period
    ret_1m = price.pct_change(21)   # 1 Month
    ret_3m = price.pct_change(63)   # 3 Months
    ret_6m = price.pct_change(126)  # 6 Months
    ret_12m = price.pct_change(252) # 12 Months
    
    # Calculate weighted sum and divide by 4
    momentum = (12 * ret_1m + 4 * ret_3m + 2 * ret_6m + ret_12m) / 4
    
    return momentum

def coppock_ts_momentum(price):
    """
    Calculates Coppock momentum for a daily price series:
    """
    ret_11m = price.pct_change(21*11)  # 11 Months
    ret_14m = price.pct_change(21*14) # 14 Months
    momentum = SMA(ret_11m+ret_14m,10)
    
    return momentum

def find_optimal_d(series):
    results = []
    # Test d from 0 (original price) to 1 (standard log returns)
    for d in np.linspace(0, 1, 11):
        fd = frac_diff(series, d) # Uses the frac_diff function from earlier
        
        # Run ADF Test
        # Null Hypothesis (H0): The series has a unit root (non-stationary)
        # Alternative Hypothesis (H1): The series is stationary
        adf_stat, p_val, _, _, _, _ = adfuller(fd.dropna())
        
        results.append({'d': round(d, 1), 'p_value': p_val})
        
    return pd.DataFrame(results)

def dev2mean(x,n=2520):
    k=int(n/2)
    #return (x - x.rolling(window=n, min_periods=k).mean()) / x.rolling(window=n, min_periods=k).std()
    return (x - x.rolling(window=n, min_periods=k).mean()) / x.rolling(window=n, min_periods=k).std()

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))    

def ohlcinterleave(df_rd,df_rn,ratio=6.5):
    ts_d = df_rd*100 #returns session from O to C
    ts_n = df_rn*100*ratio #returns overnight, from C_t-1 to O
    #ratio=6.5 is SD ratio between rd and rn
    
    #drop 1 obs so both series are same length
    ts_d = ts_d[1:]
    # Interleave the values
    combined_values = np.vstack((ts_n.values, ts_d.values)).flatten('F')
    # Interleave the index (repeats each date twice)
    new_index = np.repeat(ts_n.index, 2)
    # Create the new Series
    ts_com = pd.Series(combined_values, index=new_index)
    return ts_com
