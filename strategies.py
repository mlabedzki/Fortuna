import pandas as pd
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm
from . import analytics as an
from . import utils as ut
from . import spectral as spec

class Portfolio:

    def __init__(self, weights_df: pd.DataFrame, returns_df: pd.DataFrame):
        """Initialises the portfolio with weights and returns time series.

        Both dataframes must have matching datetime indices and column names.
        """
        self.weights = weights_df.copy()
        self.returns = returns_df.copy()
        self._align_and_validate()
    
    @classmethod
    def from_single_series(
        cls, returns_series: pd.Series, asset_name: str = "Asset_1"
    ):
        """Alternative constructor to build a portfolio from a single Series.

        Weights are automatically initialized to 1.
        Uses the Series name as the asset column name, defaulting to 'Asset_1'.
        """
        # Extract the series name or fall back to a default string if it is None
        asset_name = returns_series.name if returns_series.name is not None else "Asset_1"
                
        # Convert the single series into a DataFrame
        returns_df = returns_series.to_frame(name=asset_name)

        # Create a matching weights DataFrame filled with 1s safely
        weights_df = pd.DataFrame(
            1.0, index=returns_df.index, columns=returns_df.columns
        )

        # Instantiate the class using the converted DataFrames
        return cls(weights_df, returns_df)
    
    def _align_and_validate(self):
        """Ensures index and columns match perfectly between both dataframes."""
        # Convert indices to datetime if they aren't already
        self.weights.index = pd.to_datetime(self.weights.index)
        self.returns.index = pd.to_datetime(self.returns.index)

        # Find overlapping dates and assets
        common_dates = self.weights.index.intersection(self.returns.index)
        common_assets = self.weights.columns.intersection(
            self.returns.columns
        )

        # Filter and align both dataframes
        self.weights = self.weights.loc[common_dates, common_assets]
        self.returns = self.returns.loc[common_dates, common_assets]

    def __getattr__(self, name: str):
        """Intercepts dynamic attribute calls to treat them as asset selection.

        Returns a new Portfolio instance containing only the selected asset.
        """
        # Validate that the requested attribute name actually exists in our asset columns
        if name in self.returns.columns:
            # Extract the columns as single-column DataFrames to preserve matrix structure
            selected_weights = self.weights[[name]]
            selected_returns = self.returns[[name]]

            # Return a brand new single-asset Portfolio instance
            return Portfolio(selected_weights, selected_returns)

        # Fallback to standard Python error handling if the attribute doesn't exist anywhere
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute or asset column named '{name}'"
        )
        
    def calculate_portfolio_returns(self) -> pd.Series:
        """Calculates the total daily portfolio return."""
        # Element-wise multiplication of weights and returns, then sum across assets
        daily_returns = rel2log((self.weights*log2rel(self.returns)).sum(axis=1))
        return daily_returns

    def calculate_portfolio_vols_nocorr(self,n=21) -> pd.Series:
        """Calculates the daily weighted avarege ewma vol with 0 correlation assumption"""
        # Element-wise multiplication of weights and vols, then sum across assets
        df_vols = an.emavol(self.returns,n)
        df_vols = (self.weights*df_vols.shift(1)).sum(axis=1)
        return df_vols
    
def levret(a,r):
    if isinstance(r, Portfolio):
        r.weights = a*r.weights
        return r
    else:
        return np.log(1+a*(np.exp(r)-1))

def log2rel(x):
    if isinstance(x, Portfolio):
        x.returns = np.exp(x.returns)-1
        return x
    else:    
        return np.exp(x)-1

def rel2log(x):
    if isinstance(x, Portfolio):
        x.returns = np.log(x.returns+1)
        return x
    else:    
        return np.log(x+1)

def mm2yy(x,periods=12):
    return log2rel(x.cumsum().diff(periods))

def convol(obj,target=0.15,maxlev=10,prec=2,port=False):
    if isinstance(obj, Portfolio):
        df_returns = obj.returns
    else:
        df_returns = obj
    hv21 = an.emavol(df_returns,21,1)
    multipliers = np.round(np.minimum(maxlev,target/np.sqrt(252)/hv21.shift(1)),prec)
    multipliers.iloc[0,:] = 1
    if isinstance(obj, Portfolio):
        result = Portfolio(obj.weights*multipliers,df_returns)
    elif port:
        result = Portfolio(multipliers,df_returns)
    else:
        result = levret(multipliers,df_returns)
    return result

def portmean(df_list,lev=1,custom_weights=None,port=False):
    if any(isinstance(obj, Portfolio) for obj in df_list):
        # Access internal attributes
        n = len(df_list)
        weights=[]
        returns=[]
        for obj in df_list:
            if isinstance(obj, Portfolio):
                weights.append(obj.weights)
                returns.append(obj.returns)
            else:
                obj_weights = obj.copy()
                obj_weights[:] = 1
                weights.append(obj_weights)
                returns.append(obj)
        weights = pd.concat(weights).groupby(level=0).sum()
        returns = pd.concat(returns).groupby(level=0).mean()       
        if custom_weights is not None:
            weights = weights.multiply(custom_weights, axis=1)
        else:
            weights = weights*1/n
        result = Portfolio(weights.multiply(lev,axis=0),returns) #Portfolio(weights*lev,returns)
    else:
        y = pd.concat(df_list,axis=1)
        n = len(df_list)
        weights = y.copy()
        weights[:] = 1        
        if custom_weights is not None:
            weights = weights.multiply(custom_weights, axis=1)
        else:
            weights = weights*1/n
        if port:
            result = Portfolio(weights.multiply(lev,axis=0),y)            
        else:
            if custom_weights is not None:
                weights = custom_weights
            else:
                weights = 1/n
            result = pd.Series(rel2log(np.nansum(log2rel(y).multiply(weights, axis=1).multiply(lev,axis=0),axis=1)),index=y.index)
    return result

def splice(obj1,control,cost=0.0000,dir="long",port=False):
    if isinstance(obj1, Portfolio):
        if not isinstance(obj1, Portfolio):
            obj1 = Portfolio.from_single_series(obj1)
        returns1 = obj1.returns
    else:
        returns1 = obj1
    df = pd.concat([returns1, control], axis=1)
    if not isinstance(obj1, Portfolio):
        df.columns = ["returns1","control"]
    else:
        df.columns = list(obj1.returns.columns)+["control"]
    df.dropna(inplace=True)
    if dir=="both":
        pos1 = pd.Series(np.where(df.control <= 0, -1, 1),index=df.index)
        cost = cost*2
    if dir=="long":    
        pos1 = pd.Series(np.where(df.control <= 0, np.nan, 1),index=df.index)
    if dir=="short":
        pos1 = pd.Series(np.where(df.control <= 0, -1, np.nan),index=df.index)    
    poschanges = pos1.fillna(0).diff()!=0
    del df["control"]
    if isinstance(obj1, Portfolio):
        df_returns = df
        df_returns.columns = list(obj1.returns.columns)
        df_weights = obj1.weights.multiply(pos1, axis=0)
        result = Portfolio(df_weights, df_returns)
    elif port:
        df_returns = df
        df_returns.columns = [obj1.name]
        df_weights = pos1.to_frame()
        df_weights.columns = df_returns.columns
        result = Portfolio(df_weights, df_returns)   
    else:
        result = pos1*df.returns1 - cost*poschanges #TBD: apply cross sectional formula as we sum logrets here
    return result

def switcher(obj1,obj2,control,cost=0.0000,port=False):
    if isinstance(obj1, Portfolio) or isinstance(obj2, Portfolio):
        if not isinstance(obj1, Portfolio):
            obj1 = Portfolio.from_single_series(obj1)
        if not isinstance(obj2, Portfolio):
            obj2 = Portfolio.from_single_series(obj2)
        returns1 = obj1.returns
        returns2 = obj2.returns
    else:
        returns1 = obj1
        returns2 = obj2
    df = pd.concat([returns1, returns2, control], axis=1)
    if not isinstance(obj1, Portfolio) and not isinstance(obj2, Portfolio):
        df.columns = ["returns1","returns2","control"]
    else:
        df.columns = list(obj1.returns.columns)+list(obj2.returns.columns)+["control"]
    df.dropna(inplace=True)
    pos1 = pd.Series(np.where(df.control > 0, 1, 0),index=df.index)
    pos2 = pd.Series(np.where(df.control > 0, 0, 1),index=df.index)
    poschanges = pos1.diff()!=0
    del df["control"]
    if isinstance(obj1, Portfolio) or isinstance(obj2, Portfolio):
        df_returns = df
        df_returns.columns = list(obj1.returns.columns)+list(obj2.returns.columns)
        pos1 = obj1.weights.multiply(pos1, axis=0)
        pos2 = obj2.weights.multiply(pos2, axis=0)
        df_weights = pd.concat([pos1, pos2], axis=1)
        result = Portfolio(df_weights, df_returns)
    elif port:
        df_returns = df
        df_returns.columns = [obj1.name,obj2.name]
        df_weights = pd.concat([pos1, pos2], axis=1)
        df_weights.columns = df_returns.columns
        result = Portfolio(df_weights, df_returns)   
    else:
        result = pos1*df.returns1 + pos2*df.returns2 - cost*poschanges #TBD: apply cross sectional formula as we sum logrets here
    return result

def cooc(opens,closes,costs=0.0001,cutoff=1,strength=10000,prec=2,ret=0,sw=0,inv=0):
    # opens and closes are DF with open or close of many asset prices as wide table
    # function works on levels and implements strategy from this paper: 
    # "Overnight-Intraday Reversal Everywhere", Chun Liu et al. 2016  
    k = opens.shape[1]-opens.isnull().sum(axis=1)
    CO_ret = np.log(opens/closes.shift(1))
    OC_ret = closes/opens-1
    #OO_ret = np.log(opens/opens.shift(1))
    OC_diff = closes-opens
    if inv==1:
        CO_ret.iloc[:,0] = -CO_ret.iloc[:,0]
        OC_ret.iloc[:,0] = -OC_ret.iloc[:,0]
    #CO_ret_fut = np.column_stack([futures.ix[1:,0].values/futures.ix[:-1,1].values,futures.ix[1:,2].values/futures.ix[:-1,3].values,futures.ix[1:,4].values/futures.ix[:-1,5].values,futures.ix[1:,6].values/futures.ix[:-1,7].values,futures.ix[1:,8].values/futures.ix[:-1,9].values,]) #does not use index
    #CO_ret_fut = np.vstack([np.NaN*np.arange(0,5),CO_ret_fut])
    #CO_ret_fut = pd.DataFrame(CO_ret_fut, index=futures.index)  
    #CO_ret_fut = pd.concat([futures.ix[:,0]/futures.ix[:,1].shift(1),futures.ix[:,2]/futures.ix[:,3].shift(1),futures.ix[:,4]/futures.ix[:,5].shift(1),futures.ix[:,6]/futures.ix[:,7].shift(1),futures.ix[:,8]/futures.ix[:,9].shift(1)],axis=1)-1   
    #OC_ret_fut2 = futuresc.shift(1)/futureso.shift(1)-1
    #weightOC_fut = 10000*OC_ret_fut2.sub(OC_ret_fut2.mean(axis=1), axis=0).div(-k_fut, axis=0)
    #portOCCO_fut = CO_ret_fut*weightOC_fut - abs(weightOC_fut.diff(1)).div(1/fut_cost,axis=1)
    #OO_ret_fut = np.log(futureso/futureso.shift(1))
    #weightOO_fut = np.round(300*OO_ret_fut.sub(OO_ret_fut.mean(axis=1), axis=0).div(-k_fut, axis=0),2)
    #portOOOC_fut = OC_ret_fut*weightOO_fut - abs(weightOO_fut.diff(1)).div(1/fut_cost,axis=1)
    if sw==0:
        switch = 1
    else:
        switch = sw*(np.sign(OC_ret.mean(axis=1))+sw)/2
    weightCO = np.minimum(cutoff,np.maximum(-cutoff,np.round(strength*CO_ret.sub(CO_ret.mean(axis=1), axis=0).div(-k, axis=0),prec)))
    #portCOOC = OC_ret*weightCO - abs(weightCO).div(1/costs,axis=1)
    rel_weights = np.round((1-opens.div(opens.sum(axis=1),axis=0))*weightCO,2)
    portCOOC = OC_ret*weightCO - (abs(weightCO)/opens).div(1/costs,axis=1)
    portCOOConDIFF = (OC_diff*rel_weights) - abs(rel_weights)*costs
    
    if ret==0:
        out = an.SR(portCOOConDIFF.mean(axis=1))
    else:
        if ret==1:
            out = switch*portCOOConDIFF.sum(axis=1)
        else:
            if ret==2:
                out = portCOOC.mean(axis=1) #weightCO
            else:            
                out = weightCO #weightCO
    return out

def rcooc(opens,closes,highs,lows,costs=0.0001,cutoff=1,strength=10000,prec=2,ret=0,sw=0,inv=0):
    # opens and closes are DF with open or close of many asset prices as wide table
    # function works on loglevels, i.e. with returns, and implements strategy from this paper:
    # "Overnight-Intraday Reversal Everywhere", Chun Liu et al. 2016        
    k = opens.shape[1]-opens.isnull().sum(axis=1)
    CO_ret = np.log(opens/closes.shift(1))
    OC_ret = closes/opens-1
    #OO_ret = np.log(opens/opens.shift(1))
    #OC_diff = closes-opens
    if inv==1:
      CO_ret.iloc[:,0] = -CO_ret.iloc[:,0]
    OC_ret.iloc[:,0] = -OC_ret.iloc[:,0]
    if sw==0:
      switch = 1
    else:
      switch = sw*(np.sign(OC_ret.mean(axis=1))+sw)/2
    weightCO = np.minimum(cutoff,np.maximum(-cutoff,np.round(strength*CO_ret.sub(CO_ret.mean(axis=1), axis=0).div(-k, axis=0),prec)))
    #portCOOC = OC_ret*weightCO - abs(weightCO).div(1/costs,axis=1)
    rel_weights = np.round((1-opens.div(opens.sum(axis=1),axis=0))*weightCO,2)

    #portCOOC = OC_ret*weightCO - (abs(weightCO)/opens).div(1/costs,axis=1)
    if ret<2:
      lowsy = (lows-opens) #.clip(upper=0)
    highsy = (highs-opens) #.clip(upper=0)
    if ret==2:
      lowsy = (lows/opens-1) #.clip(upper=0)
    highsy = (highs/opens-1) #.clip(upper=0)
    long0short1 = pd.concat([lowsy.ix[:,0],highsy.ix[:,1]],axis=1) # lowhigh
    long1short0 = pd.concat([highsy.ix[:,0],lowsy.ix[:,1]],axis=1) # highlow
    #portCOOConDIFF = (OC_diff*rel_weights) - abs(rel_weights)*costs
    loses0 = pd.Series(np.where(rel_weights.ix[:,0] <= 0, long1short0.ix[:,0], long0short1.ix[:,0]),index=long1short0.index)
    loses1 = pd.Series(np.where(rel_weights.ix[:,1] <= 0, long0short1.ix[:,1], long1short0.ix[:,1]),index=long1short0.index)
    oldnames = opens.columns
    loses = pd.concat([loses0,loses1],axis=1)
    loses.columns = oldnames 
    if ret<2:
      loses = loses*rel_weights - abs(rel_weights)*costs
    if ret==2:    
      loses = loses*weightCO - (abs(weightCO)/opens).div(1/costs,axis=1)

    if ret==0:
      out = loses
    else:
      if ret==1:
          out = switch*loses.sum(axis=1)
      else:
          if ret==2:
              out = loses.mean(axis=1) #weightCO
          else:            
              out = weightCO #weightCO
    return out    

def dailycc(x,costs=0.0001,cutoff=1,strength=10000,prec=2,ret=0,sw=0):
    #function implements strategy from the same paper that COOC, but from 1day close to next close
    #k = x.shape[1]-x.isnull().sum(axis=1)
    k = x.shape[1]

    closes = pd.DataFrame(x, index=pd.to_datetime(x.index))
    closes = pd.concat([closes,pd.Series(closes.index.dayofweek, index=closes.index)],axis=1)
    columns = closes.columns.tolist()
    columns = columns[0:x.shape[1]]
    columns.append('dw')
    closes.columns = columns
    #y = closes.loc[(closes['dw'] == 0) | (closes['dw'] == 3)]
    opens = x.loc[(closes['dw'] == 0)]
    opens.index = opens.index.shift(3,'D')
    closes = x.loc[(closes['dw'] == 3)]
    #opens = x.loc[(closes['dw'] == 3)]
    #opens.index = opens.index.shift(4,'D')
    #closes = x.loc[(closes['dw'] == 0)]
    y = pd.concat([opens,closes],axis=1)
    y.dropna(inplace=True)
    opens = y.ix[:,0:2]
    closes = y.ix[:,2:4]

    CO_ret = np.log(opens/closes.shift(1))
    OC_ret = closes/opens-1
    #OO_ret = np.log(opens/opens.shift(1))
    if sw==0:
        switch = 1
    else:
        switch = sw*(np.sign(OC_ret.mean(axis=1))+sw)/2
    weightCO = np.minimum(cutoff,np.maximum(-cutoff,np.round(strength*CO_ret.sub(CO_ret.mean(axis=1), axis=0).div(-k, axis=0),prec)))
    portCOOC = OC_ret*weightCO - abs(weightCO).div(1/costs,axis=1)
    if ret==0:
        out = an.SR(portCOOC.mean(axis=1),b=52)
    else:
        if ret==1:
            out = switch*portCOOC.mean(axis=1)
        else:
            out = weightCO
    return out

def quadratic_utility_strategy(open_ts, close_ts, gamma=3, initial_window=252):
    # from paper: "The Day Destroys the Night, Night Extends the Day" by Lou et al. 2024
    """Constructs the 'Quadratic utility mean-and-vol-forecast based' strategy.

    Parameters:
    - open_ts, close_ts: pd.Series of S&P 500 daily Open and Close prices.
    - gamma: Relative risk aversion coefficient (default = 3).
    - initial_window: Number of days to seed the expanding window regression.
    """
    # 1. Calculate returns
    # Daily market excess return (Close-to-Close) - used as the target to forecast
    # For a pure strategy, we use raw returns (assuming 0 cash rate as baseline)
    r_market = close_ts.pct_change()

    # Daily Close-to-Open (Overnight) return
    r_overnight = open_ts / close_ts.shift(1) - 1

    # 2. Smooth the overnight return signal using EWMA (e.g., 20-day half-life)
    # The paper utilizes a smoothed signal to capture persistent structural flows
    ewma_overnight = r_overnight.ewm(halflife=20).mean()

    # 3. Calculate rolling daily proxy for volatility (e.g., 21-day rolling variance)
    # This serves as the target variable for the volatility forecasting model
    vol_target = r_market.rolling(window=21).var()

    # Align data into a single DataFrame and drop NaNs
    df = pd.DataFrame(
        {
            "r_market": r_market,
            "vol_target": vol_target,
            "signal": ewma_overnight.shift(
                1
            ),  # Shift by 1 day to ensure out-of-sample forecast
        }
    ).dropna()

    # Arrays to store out-of-sample predictions
    mean_forecasts = np.zeros(len(df))
    vol_forecasts = np.zeros(len(df))

    # 4. Out-of-sample expanding window regressions
    # Loop over time step t to forecast t+1 using only data up to step t
    for i in range(initial_window, len(df)):
        # Define historical training subsets
        train_df = df.iloc[:i]

        X_train = sm.add_constant(train_df["signal"])
        X_next = np.array([1.0, df["signal"].iloc[i]])

        # A. Forecast the Mean (OLS)
        y_mean_train = train_df["r_market"]
        model_mean = sm.OLS(y_mean_train, X_train).fit()
        mean_forecasts[i] = np.dot(X_next, model_mean.params)

        # B. Forecast the Volatility / Variance (OLS)
        y_vol_train = train_df["vol_target"]
        model_vol = sm.OLS(y_vol_train, X_train).fit()
        # Ensure variance projection remains strictly positive
        vol_forecasts[i] = max(1e-6, np.dot(X_next, model_vol.params))

    # Add forecasts to main DataFrame
    df["hat_r"] = mean_forecasts
    df["hat_sigma2"] = vol_forecasts

    # Filter out the initial warmup window for backtest evaluation
    strategy_df = df.iloc[initial_window:].copy()

    # 5. Apply the Allocation Formula: w = hat_r / (gamma * hat_sigma2)
    # No shorting or leverage constraints applied, matching the paper's specific setting
    strategy_df["weight"] = strategy_df["hat_r"] / (
        gamma * strategy_df["hat_sigma2"]
    )

    # 6. Calculate performance metrics
    strategy_df["strategy_return"] = (
        strategy_df["weight"] * strategy_df["r_market"]
    )

    return strategy_df[["r_market", "weight", "strategy_return"]]

def domestic(returns,base2quote,cost=0.00025):#base2quote is for example USDPLN
    x = np.log(1+returns)
    fAcc=np.exp(np.cumsum(x))
    z=fAcc.diff(1)
    z[2]=1
    m = z.copy().to_frame().join(base2quote)
    return np.cumsum(m.ix[:,0]*m.ix[:,1]*(1-cost))

def shortMRstrat(returns, idx, m, up, lo, spread=0.2, v=0):
    # Ensure inputs are Series
    idx = pd.Series(idx)
    returns = pd.Series(returns)
    
    # 1. Define signal conditions
    shortenter = idx > (m + up)
    shortclose = idx < (m - lo)
    
    # 2. Replicate R's ifelse(shortenter, 2, 1) logic
    enter_val = shortenter.astype(int) + 1
    
    # 3. Identify first (v=1) or last (v=0) occurrence
    if v == 1:
        tshortsignals = np.floor((1 + enter_val.diff()) / 2)
    else:
        tshortsignals = np.floor((1 - enter_val.diff()) / 2)
        
    # 4. Closing signals (R's lag -1 is Python's shift -1)
    close_val = shortclose.astype(int) + 1
    tcloseshortsignals = (np.floor((1 + close_val.diff()) / 2) * 999).shift(-1)
    
    # 5. Position tracking using your custom helpers
    # capcumsum is called here
    raw_signals = (tshortsignals - tcloseshortsignals).shift(1).fillna(0)
    shorts = capcumsum(raw_signals)
      
    res = splice(levret(1,returns), shorts.shift(1), cost=0.000, dir="long")
    
    # 6. Apply the spread penalty
    spread_impact = (np.abs(np.sign(shorts.shift(1)).diff()) * spread) / 2
    
    return res - spread_impact

def capcumsum(x_input):
    # Handle pandas Series/DataFrame or numpy array
    if isinstance(x_input, (pd.Series, pd.DataFrame)):
        x = x_input.values
    else:
        x = np.asarray(x_input)
    
    # Ensure x is 1D or 2D for indexing (matrix logic)
    original_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        
    n = x.shape[0]
    # Initialize output array with NaNs
    y = np.full(x.shape, np.nan)
    
    # Handle the first row (na20 should ensure this isn't NaN usually)
    y[0] = x[0]
    
    # Iterative logic: y[i] = max(0, x[i] + y[i-1])
    # This prevents the cumsum from going negative
    for i in range(1, n):
        y[i] = np.maximum(0, x[i] + y[i-1])
    
    # Return to original format (Pandas or Numpy)
    if isinstance(x_input, pd.Series):
        return pd.Series(y.flatten(), index=x_input.index)
    elif isinstance(x_input, pd.DataFrame):
        return pd.DataFrame(y, index=x_input.index, columns=x_input.columns)
    
    return y.reshape(original_shape)

def run_minmax_strategy(df, window=20, minmax=10, num_std=0.35, cost=0.001):
    name = df.name
    df = pd.DataFrame(df)
    #df['MA'] = ZLEMA(df[name], window)
    df['MA'] = an.KAMA(df[name])
    masigma = an.smavol(df['MA'].diff(),252,1)
    df['bull'] = (df['MA'].diff()>0) & (df['MA'] > num_std*masigma + df['MA'].rolling(window=minmax, center=False).min().shift(1))
    df['bear'] = (df['MA'].diff()<0) & (df['MA'] < -num_std*masigma + df['MA'].rolling(window=minmax, center=False).max().shift(1))
    df['sent'] = df['bull'].astype(float) - df['bear'].astype(float)
    df['sent'] = df['sent'].replace(0, np.nan)
    df['sent'] = df['sent'].ffill()
    return df['sent']
    
def run_bb_strategy(df, window=20, num_std=2, cost=0.001, mod=None, num_std_lo=None, noema=False):
    # calculate Bollinger Bands and strategy returns without loops
    name = df.name
    df = pd.DataFrame(df)
    df['logrets'] = np.log(df[name]).diff()
    
    if num_std_lo is None:
        num_std_lo = num_std
    if mod is None:
        mod = 1 
        
    # 1. Calculate Rolling Mean and Standard Deviation
    if noema:
        df['sma'] = df[name].rolling(window=window).mean()
        df['std'] = df[name].rolling(window=window).std()
    #alternative:
    else:
        df['sma'] = an.EMA(df[name],window)
        df['std'] = an.smavol(df[name].diff(),window,1)
    
    # 2. Define Upper and Lower Bands
    df['upper'] = df['sma'] + (mod * num_std * df['std'])
    df['lower'] = df['sma'] - (mod * num_std_lo * df['std'])
    
    # 3. Vectorized Signal Generation
    # Long (1) if price < lower band; Short (-1) if price > upper band
    df['signal'] = 0
    df.loc[df[name] < df['lower'], 'signal'] = 1
    df.loc[df[name] > df['upper'], 'signal'] = -1
    
    # 4. Position Handling (Forward Fill)
    # Stays in position until an opposite signal is triggered
    df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
    
    # 5. Calculate Returns (Shift position to avoid look-ahead bias)
    # The return at time t is based on the position held at the end of t-1
    df['strat_rets'] = df['position'].shift(1) * df['logrets']
    
    # 6. Apply Transaction Costs
    # Cost occurs on the absolute change in position (entry, exit, or flip)
    df['trades'] = df['position'].diff().abs().fillna(0)
    df['strat_rets_tc'] = df['strat_rets'] - (df['trades'] * cost)

    return df['position']

def run_bbconfirm_strategy(df, window=20, num_std=2, cost=0.001, mod=None, num_std_lo=None, noema=False):
    # calculate Bollinger Bands and strategy returns without loops
    # in this version we wait for signal confirmation and exit ASAP, thus short/long assymetry
    name = df.name
    df = pd.DataFrame(df)
    df['logrets'] = np.log(df[name]).diff()
    
    if num_std_lo is None:
        num_std_lo = num_std
    if mod is None:
        mod = 1 
        
    # 1. Standard BB Calculation
    if noema:
        df['sma'] = df[name].rolling(window=window).mean()
        df['std'] = df[name].rolling(window=window).std()
    #alternative:
    else:
        df['sma'] = an.EMA(df[name],window)
        df['std'] = an.smavol(df[name].diff(),window,1)       
    
    df['upper'] = df['sma'] + (mod * num_std * df['std'])
    df['lower'] = df['sma'] - (mod * num_std_lo * df['std'])

    # 2. Define Crossing Logic
    # Price crosses Upper from above: yesterday > upper AND today < upper
    short_entry = (df[name].shift(1) > df['upper'].shift(1)) & (df[name] < df['upper'])
    
    # Price crosses Lower from above: yesterday > lower AND today < lower
    short_exit = (df[name].shift(1) > df['lower'].shift(1)) & (df[name] < df['lower'])
    
    # Price crosses Lower from below: yesterday < lower AND today > lower
    long_entry = (df[name].shift(1) < df['lower'].shift(1)) & (df[name] > df['lower'])
    
    # Price crosses Upper from below: yesterday < upper AND today > upper
    long_exit = (df[name].shift(1) < df['upper'].shift(1)) & (df[name] > df['upper'])

    # 3. Vectorized State Machine
    # We create separate columns for long and short states
    df['long_state'] = np.nan
    df.loc[long_entry, 'long_state'] = 1
    df.loc[long_exit, 'long_state'] = 0
    df['long_pos'] = df['long_state'].ffill().fillna(0)

    df['short_state'] = np.nan
    df.loc[short_entry, 'short_state'] = -1
    df.loc[short_exit, 'short_state'] = 0
    df['short_pos'] = df['short_state'].ffill().fillna(0)

    # 4. Combine Positions
    # Note: If both are active, they net out.
    df['position'] = df['long_pos'] + df['short_pos']

    # 5. Performance calculation
    df['strat_rets'] = df['position'].shift(1) * df['logrets']
    df['trades'] = df['position'].diff().abs().fillna(0)
    df['strat_rets_tc'] = df['strat_rets'] - (df['trades'] * cost)

    return df['position']

def perfstrat(levels,period,begin=None,end=None,cost=0,mod=0.2): 
    base = 5
    #sma_perf = np.log(levels).rolling(window=period*base).mean().shift(-int(period*base/2))
    #sma_perf = np.log(levels).ewm(span=period*base).mean().shift(-int(period*base*mod))
    sma_perf = an.DEMA(np.log(levels),period*base).shift(-int(period*base*0.15))
    #sma_perf = ZLEMA(np.log(levels),period*base).shift(-int(period*base*0.15))
    volrel = float(ut.trimdf(an.emavol(sma_perf.diff()),begin,end).mean()/ut.trimdf(an.emavol_level(levels),begin,end).mean())
    periods = spec.calculate_periodic_times_series(np.log(levels)/sma_perf,1)
    t1 = splice(np.log(levels).diff().shift(-1),(sma_perf-sma_perf.shift()),cost=cost)
    t2 = splice(np.log(levels).diff().shift(-1),(np.log(levels)<sma_perf),cost=cost)
    return (period,an.SR(ut.trimdf(t1,begin,end)),an.SR(ut.trimdf(t2,begin,end)),volrel,periods,int(period*base*mod))

def meanstrat(pclose,popen,base,par,start="2005-10-01"): 
    popen = popen[popen.index >= start]
    pclose = pclose[pclose.index >= start]
    mean = pclose.rolling(window=par*base).mean()
    mrel = pclose/mean-1
    fmrel = mrel.shift(-base)
    freturn = log2rel(np.log(pclose.shift(-base))-np.log(popen.shift(-1)))
    return an.SR(freturn*np.sign(fmrel),b=252/base)

def distill(returns,overlay=None,bins=12,delay=1,m=None,si=None,mod=None):
    if overlay is None:
        overlay = returns.copy()
    returns2d = pd.concat([overlay,returns.shift(-delay)],axis=1)
    returns2d = returns2d.dropna()
    returns2d.columns = ['returns','returnsfut']
    mu, sigma = sps.norm.fit(returns2d.returns)
    if m is None:
        m = mu
    if si is None:
        si = sigma
    returns2d.returns = (returns2d.returns-m)/si
    returns2d = returns2d.sort_values(by='returns', ascending=True)
    binsize = int(len(returns2d)/bins)
    if mod is None:
        mod=0
    modi = int(binsize*mod)
    out = []
    bincuts = []
    for i in range(bins):
        end = min(((i+1)*binsize+modi),len(returns2d))
        out.append(returns2d.iloc[(i*binsize+modi) : end,:])
        bincuts.append(returns2d.iloc[end-1,0])
    bins = {"cuts": np.array(bincuts), "m": m, "si": si}
    return (out,bins)

def assignbin(nreturns,bincuts,mode="full"):#if bincuts would not be known apriori, then possibility is to use pd.cut(s, bins=bins)
    out = pd.DataFrame(nreturns.copy())
    bincuts[len(bincuts)-1]=99
    xx = pd.Series(np.searchsorted(bincuts, nreturns),index=nreturns.index)
    if mode=="full":
        for i,b in enumerate(bincuts):
            out['b'+str(i)] = (i==xx).astype(int)
    else:
        out['bin'] = xx
    return out
    
def testbinstrats(rwig20bins,bins,start=None,end=None):
    out=[]
    for i, (b,cut) in enumerate(zip(rwig20bins,bins['cuts'])):
        #stra = ut.trimdf(np.sign(float(cut))*b.returnsfut,start,end)
        #stra = ut.trimdf(np.sign((b.returns))*b.returnsfut,start,end)
        stra = ut.trimdf(np.sign(np.sign(b.returns).mean())*b.returnsfut,start,end)
        out.append((float(cut),an.SR(stra),len(b)))#,"2008-01-01","2015-01-01"))
    out = pd.DataFrame(out)
    out.columns=["cut","SR","size"]
    return out

def testbinstratsoos(oosreturns,bins,overlay=None,delay=1,start=None,end=None):
    if overlay is None:
        overlay = oosreturns.copy()   

    # returns2d = pd.concat([overlay,oosreturns.shift(-delay)],axis=1)
    # returns2d = returns2d.dropna()
    # returns2d.columns = ['returns','returnsfut']

    # returns2d.returns = (returns2d.returns-bins['m'])/bins['si']
    # overlay = assignbin(overlay,bins['cuts'],mode=0)
    # overlay['returnsfut'] = oosreturns.shift(-1)
        
    overlay = (overlay-bins['m'])/bins['si']
    overlay.name = 'returns'
    overlay = assignbin(overlay,bins['cuts'],mode=0)
    overlay['returnsfut'] = oosreturns.shift(-delay)
    
    out=[]
    for i in range(len(overlay.bin.unique())):
        b = overlay[overlay.bin==i]
        cut = bins['cuts'][i]
        #stra = ut.trimdf(np.sign(float(cut))*b.returnsfut,start,end)
        #stra = ut.trimdf(np.sign((b.returns))*b.returnsfut,start,end)
        stra = ut.trimdf(np.sign(np.sign(b.returns).mean())*b.returnsfut,start,end)
        out.append((float(cut),an.SR(stra),len(b)/len(overlay)))#,"2008-01-01","2015-01-01"))
    out = pd.DataFrame(out)
    out.columns=["cut","SR","size"]
    return out