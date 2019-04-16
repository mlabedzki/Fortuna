import pandas as pd
import numpy as np

def maxdd_abs(eq,stress=None):
    dd = eq.div(eq.cummax()).sub(1)
    if stress is not None:
        dd += stress
    mdd = dd.min()
    end = dd.argmin()
    start = eq.loc[:end].argmax()
    return mdd, start, end
    
def maxdd_rel(returns,stress=None):
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    if stress is not None:
        dd += stress    
    mdd = dd.min()
    end = dd.argmin()
    start = r.loc[:end].argmax()
    return mdd, start, end

def SR(x,b=252):
    y = np.log(1+x)
    return np.nanmean(y*b)/(np.nanstd(y)*np.sqrt(b))
    
def splot(signal,control):
    dane = pd.concat([signal,control], axis=1)
    dane.columns = ["returns","control"]
    newsignal = pd.Series(np.where(dane['control'] <= 0, 0, dane['returns']),index=dane.index)
    return newsignal
    
def domestic(returns,usdpln):
    x = np.log(1+returns)
    fAcc=np.exp(np.cumsum(x))
    z=fAcc.diff(1)
    z[2]=1
    m = z.copy().to_frame().join(usdpln)
    return np.cumsum(m.ix[:,0]*m.ix[:,1]*(1-0.00025))

def emavol(returns,periods=63,pinyear=252):
    return np.sqrt(pd.ewma(returns**2,span=periods)*pinyear)
