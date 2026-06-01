# fortuna
fortuna is a python library for quantitative investment portfolio strategies analysis

Please follow below exemplary analysis to be introduced to some of its concepts.

``` Python
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(r'C:\Users\Mikolaj Labedzki\Documents\Python')
import fortuna as ft
import sklearn.model_selection as sklms
```

We will work on below date range:

``` Python
begin = '2000-09-19'
end = '2026-04-30'
```

and use these tickers: ^GSPC for S&P 500 and GC=F for Gold Futures
``` Python
tickers = ['ES=F', 'NQ=F', 'GC=F', 'HG=F','ZN=F']
macroindicators = ['DBAA','DAAA','DGS30','T10Y2Y','VIXCLS','DTB3']
```

Then, we fetch data for the last 30 years
``` Python
df_data = yf.download(tickers, period='30y')
df_data = df_data.ffill()
df_returns = np.log(df_data['Close']).diff()
df_returns = ft.trimdf(df_returns,begin,end)
df_returns.columns = [s.replace('=','_').replace('.','_').replace('-','_') for s in df_returns.columns]
df_data = ft.trimdf(df_data,begin,end)
df_closes = df_data['Close']
df_closes.columns = [s.replace('=','_').replace('.','_').replace('-','_') for s in df_closes.columns]
df_returnscv = ft.convol(df_returns,target=0.2,maxlev=5)
df_returnscvp = ft.convol(df_returns,target=0.2,maxlev=5,port=True) #same but using portfolio object

df_fred_d = ft.load_fred(macroindicators,start=begin,end=end)
df_fred_d = df_returns.GC_F.to_frame().join(df_fred_d).iloc[:,1:] #align indicators to market data
df_fred_d = df_fred_d.ffill()
```

#split sets for training and testing
``` Python
retscv_train, retscv_test, rets_train, rets_test = sklms.train_test_split(df_returnscv, df_returns, test_size=0.5, shuffle=False)
```

for portfolio object splitting has to be done in 2 steps:
``` Python
retscvpw_train, retscvpw_test, retscvpr_train, retscvpr_test, = sklms.train_test_split(df_returnscvp.weights, df_returnscvp.returns, test_size=0.5, shuffle=False)
retscvp_train = ft.Portfolio(retscvpw_train,retscvpr_train)
retscvp_test = ft.Portfolio(retscvpw_test,retscvpr_test)
```

now, some basic performance analysis
``` Python
ft.strat_stats(rets_train.ES_F)
ft.strat_stats(rets_train.NQ_F)
ft.strat_stats(rets_train.GC_F)
ft.strat_stats(rets_train.HG_F)
ft.strat_stats(rets_train.ZN_F)

plt.plot(rets_train.ES_F.cumsum())
plt.plot(rets_train.NQ_F.cumsum())
plt.plot(rets_train.GC_F.cumsum())
plt.plot(rets_train.HG_F.cumsum())
plt.plot(rets_train.ZN_F.cumsum())
```

we test also constant volatility strategies:
``` Python
ft.strat_stats(retscv_train.ES_F)
ft.strat_stats(retscv_train.NQ_F)
ft.strat_stats(retscv_train.GC_F)
ft.strat_stats(retscv_train.HG_F)
ft.strat_stats(retscv_train.ZN_F)

plt.plot(retscv_train.ES_F.cumsum())
plt.plot(retscv_train.NQ_F.cumsum())
plt.plot(retscv_train.GC_F.cumsum())
plt.plot(retscv_train.HG_F.cumsum())
plt.plot(retscv_train.ZN_F.cumsum())
```

then check how all instruments perform combined in normal position and cvol position
``` Python
ft.strat_stats(ft.portmean([rets_train]))
ft.strat_stats(ft.portmean([retscv_train]))
```

particularly below is nonequity part, which we can consider defensive assets
``` Python
d1=ft.portmean([rets_train.GC_F,rets_train.HG_F,rets_train.ZN_F],port=True);ft.strat_stats(d1)
d2=ft.portmean([retscvp_train.GC_F,retscvp_train.HG_F,retscvp_train.ZN_F],port=True);ft.strat_stats(d2)
```

below we test some ideas from alvarezquanttrading.com

first, TA based strategies for ES_F

``` Python
t=ft.splice(rets_train.ES_F,(ft.coppock_ts_momentum(df_closes.ES_F).diff(21*3)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(df_closes.ES_F-ft.SMA((df_closes.ES_F),200)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(np.log(df_closes.ES_F)-ft.SMA(np.log(df_closes.ES_F),200)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(df_closes.ES_F-ft.SMA((df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(np.log(df_closes.ES_F)-ft.SMA(np.log(df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(np.log(df_closes.ES_F)-ft.SMA(np.log(df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1)-0.01,dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(df_closes.ES_F/ft.SMA(df_closes.ES_F,200)).shift(1)-1-0.01,dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(np.log(df_closes.ES_F)-ft.SMA(np.log(df_closes.ES_F),200)).shift(1)-0.01,dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(ft.SMA(np.log(df_closes.ES_F),50)-ft.SMA(np.log(df_closes.ES_F),200)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(ft.SMA((df_closes.ES_F),50)-ft.SMA((df_closes.ES_F),200)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(ft.SMA((df_closes.ES_F),50)-ft.SMA((df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(ft.SMA(np.log(df_closes.ES_F),50)-ft.SMA(np.log(df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(ft.SMA(np.log(df_closes.ES_F),50)-ft.SMA(np.log(df_closes.ES_F),200)).shift(1)-0.01,dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(ft.SMA((df_closes.ES_F),50)/ft.SMA((df_closes.ES_F),200)).shift(1)-1-0.01,dir="long");ft.strat_stats(t)
```

check how best one (SR+Calmar) works with convol strat
``` Python
t=ft.splice(rets_train.ES_F,(df_closes.ES_F-ft.SMA((df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(t)
tcv=ft.splice(retscv_train.ES_F,(df_closes.ES_F-ft.SMA((df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(tcv)
plt.plot(t.cumsum())
plt.plot(tcv.cumsum())
```

check how long/short version works and how going long second asset instead of short works:
``` Python
t=ft.splice(rets_train.ES_F,(df_closes.ES_F-ft.SMA((df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1),dir="both");ft.strat_stats(t)
t=ft.switcher(rets_train.ES_F,rets_train.GC_F,(df_closes.ES_F-ft.SMA((df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1));ft.strat_stats(t)
```

second, non-TA based strategies aka exog based strats

instead of LQD and IEF funds ratio we will use following ratio: "BAA corp yields"/"AAA corp yields"
we will also add some additional ratios, since we also avoid using VWO and BND funds due to short history

``` Python
lir = df_fred_d.DBAA/df_fred_d.DAAA
cgr = df_closes.HG_F/df_closes.GC_F
gcr = df_closes.GC_F/df_closes.HG_F
sgr = df_closes.ES_F/df_closes.GC_F
sbr = df_closes.ES_F/df_closes.ZN_F
t=ft.splice(rets_train.ES_F,-(lir-ft.EMA((lir),200)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(df_fred_d.DGS30-ft.EMA(df_fred_d.DGS30,200)).rolling(window=1, center=False).min().shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,-(df_fred_d.T10Y2Y-ft.EMA(df_fred_d.T10Y2Y,200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,-(df_fred_d.DBAA-ft.EMA(df_fred_d.DBAA,200)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,-(df_fred_d.VIXCLS-ft.EMA(df_fred_d.VIXCLS,200)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,-df_fred_d.VIXCLS.shift(1)+20,dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(cgr-ft.EMA((cgr),200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(1/gcr-1/ft.EMA((gcr),200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,(sgr-ft.SMA(sgr,200)).rolling(window=5, center=False).min().shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,((ft.weighted_ts_momentum(df_closes.HG_F)>0)*(ft.weighted_ts_momentum(cgr)>0)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,((ft.weighted_ts_momentum(sgr)>0)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,((ft.weighted_ts_momentum(sbr)>0)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,((ft.weighted_ts_momentum(df_closes.ZN_F)>0)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,((ft.weighted_ts_momentum(df_closes.ZN_F)>0)*(ft.weighted_ts_momentum(sbr)>0)).shift(1),dir="long");ft.strat_stats(t)
t=ft.splice(rets_train.ES_F,((ft.weighted_ts_momentum(df_closes.GC_F)>0)).shift(1),dir="long");ft.strat_stats(t)
```

here no longer ideas from aforementioned website, we will create some portfolio and analyze it
we will select 3 best strats in different categories, this time we will call functions with port=True for full traceability
``` Python
t1=ft.splice(rets_train.ES_F,(df_closes.ES_F-ft.SMA((df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1),dir="long",port=True);ft.strat_stats(t1)
t2=ft.splice(rets_train.ES_F,(cgr-ft.EMA((cgr),200)).rolling(window=5, center=False).min().shift(1),dir="long",port=True);ft.strat_stats(t2)
t3=ft.splice(rets_train.ES_F,((ft.weighted_ts_momentum(sbr)>0)).shift(1),dir="long",port=True);ft.strat_stats(t3)
```

then see differsivication results in portfolio combining them
``` Python
p1=ft.portmean([t1,t2,t3],port=True);ft.strat_stats(p1,benchmark=rets_train.ES_F)
p2=ft.portmean([t1,t2,t3,d1],port=True);ft.strat_stats(p2,benchmark=rets_train.ES_F)
p3=ft.portmean([t1,t2,t3,d2],port=True);ft.strat_stats(p3,benchmark=rets_train.ES_F)
```

as p3 has low vol, we may look into more levared version, here we add levarage to match vol of equity market
``` Python
print(rets_train.ES_F.std()/p3.calculate_portfolio_returns().std())
print((0.2/ft.emavol(p3.calculate_portfolio_returns(),21*1,252).shift(1)).mean())
```

then we compute dynamic ratio using moving window vol, we use the former scalar as max lev, we also use asinh centralized at 1 as signal flattener
``` Python
lev_series = np.minimum(7,0.2/ft.emavol(p3.calculate_portfolio_returns(),21*1,252).shift(1))
lev_series = np.asinh(lev_series-1)+1
lev_series.iloc[0:2]=1
p3b=ft.portmean([t1,t2,t3,d2],lev=lev_series,port=True);ft.strat_stats(p3b,benchmark=rets_train.ES_F)
```

we can see exact positioning of each asset in .weights field:
``` Python
p3b.weights.describe()
p3b.weights.sum(axis=1).describe() #how total levarage changes over time
```

check how best strats work in testing period:
``` Python
to1=ft.splice(rets_test.ES_F,(df_closes.ES_F-ft.SMA((df_closes.ES_F),200)).rolling(window=5, center=False).min().shift(1),dir="long",port=True);ft.strat_stats(to1)
to2=ft.splice(rets_test.ES_F,(cgr-ft.EMA((cgr),200)).rolling(window=5, center=False).min().shift(1),dir="long",port=True);ft.strat_stats(to2)
to3=ft.splice(rets_test.ES_F,((ft.weighted_ts_momentum(sbr)>0)).shift(1),dir="long",port=True);ft.strat_stats(to3)
do1=ft.portmean([rets_test.GC_F,rets_test.HG_F,rets_test.ZN_F],port=True);ft.strat_stats(do1)
do2=ft.portmean([retscvp_test.GC_F,retscvp_test.HG_F,retscvp_test.ZN_F],port=True);ft.strat_stats(do2)
po1=ft.portmean([to1,to2,to3],port=True);ft.strat_stats(po1,benchmark=rets_test.ES_F)
po2=ft.portmean([to1,to2,to3,do1],port=True);ft.strat_stats(po2,benchmark=rets_test.ES_F)
po3=ft.portmean([to1,to2,to3,do2],port=True);ft.strat_stats(po3,benchmark=rets_test.ES_F)
```

again we want to see vol adjusted version of strategy, so again apply lev at strat level
``` Python
lev_series_o = np.minimum(7,0.2/ft.emavol(po3.calculate_portfolio_returns(),21*1,252).shift(1))
lev_series_o = np.asinh(lev_series_o-1)+1
lev_series_o.iloc[0:2]=1
po3b=ft.portmean([to1,to2,to3,do2],lev=lev_series_o,port=True);ft.strat_stats(po3b,benchmark=rets_test.ES_F)   
ft.pltcum(po1)
ft.pltcum(po2)
ft.pltcum(po3)
ft.pltcum(po3b)
ft.pltcum(rets_test.ES_F)
# vs benchmarks for this period:
ft.strat_stats(rets_test.ES_F)
ft.strat_stats(rets_test.GC_F)
```

we can observe that only t1 was better than the product it tried to outperform
while t2 & t3 underperformed, mixing t1,t2,t3 in equaly weighted portfolio is improvement over t1
gold had worse time in testing period, mixing gold with other nonequity assets is worse than gold alone
mixing nonequity assets in cvol style is even worse than the noncvol case
having said so, adding that nonequity mix to t1,t2,t3 mix (in either normal or cvol) is improving its performance
as expected, the performance in testing period although better than benchmark is not that much better as in traning period
alpha went down from 6% to 3% (in version that is not levered on top), but is still positive
