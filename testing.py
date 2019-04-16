import loaders as ld
import analytics as an
import strategies as frtn

##################
### CURRENCIES ###
##################
ccy_names = ['sekusd','mxnusd','cadusd','chfusd','jpyusd','audusd','gbpusd','eurusd','nzdusd']
ccy_cost = np.array([5.3, 4.9, 1.7, 2.1, 1.6, 2.5, 1.9, 0.8, 3.6])/10000

ccyso = ld.stacklisted(ccy_names,"d",0)[datetime(2010, 11, 15):]
ccysc = ld.stacklisted(ccy_names,"d",3)[datetime(2010, 11, 15):]

frtn.cooc(ccyo,ccyc,ccy_costs,1,900,2)
x = frtn.cooc(ccyo,ccyc,ccy_costs,1,900,2,ret=1)
plt.plot(np.cumsum(np.log(1+x)));plt.show()

#############
### PLOTS ###
#############
OC_ret_fut = frtn.cooc(ccyso,ccysc,fut_cost,1,900,2,ret=3)
CO_ret_fut = frtn.cooc(ccyso,ccysc,fut_cost,1,900,2,ret=2)
CO_ret_fut = frtn.cooc(ccysc.shift(1),ccyso,fut_cost,1,900,2,ret=3)
CO_ret_fut = ccyso/ccysc.shift(1)-1
OC_ret_fut = ccysc/ccyso-1

plt.plot(pd.rolling_corr(OC_ret_fut.ix[:,0],OC_ret_fut.ix[:,1],60));plt.show()
plt.plot(pd.rolling_corr(ccysc.ix[:,0],ccysc.ix[:,1],60));plt.show()
plt.plot(pd.rolling_mean(CO_ret_fut.std(axis=1),60));plt.show()

x = portCOOC_fut.mean(axis=1)
plt.plot(pd.rolling_mean(CO_ret_fut.std(axis=1),10))
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,y1,y2))
plt.plot(pd.rolling_mean(OC_ret_fut.std(axis=1),63))
#regr = linear_model.LinearRegression()
#results = regr.fit(pd.DataFrame(sample.ix[:,1]),pd.DataFrame(sample.ix[:,0]))
#pred = regr.predict(pd.DataFrame(sample.ix[:,1]))
#print 'Variance score: %.2f' % r2_score(pd.DataFrame(sample.ix[:,0]), pred)

sample = pd.concat([portCOOC_fut.mean(axis=1),pd.rolling_mean(OC_ret_fut.std(axis=1),63)],axis=1)
sample.dropna(axis=0, inplace=True)
model = sm.OLS(pd.DataFrame(sample.ix[:,0]),pd.DataFrame(sample.ix[:,1]))
results_fu = model.fit()
print results_fu.summary()
predictions = results_fu.predict(pd.DataFrame(sample.ix[:,1]))

plt.scatter(pd.DataFrame(sample.ix[:,1]),pd.DataFrame(sample.ix[:,0]),  color='black')
plt.plot(pd.DataFrame(sample.ix[:,1]),predictions , color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

plt.plot(pd.rolling_corr(CO_ret_fut.ix[:,0],CO_ret_fut.ix[:,2],20))
plt.plot(pd.rolling_corr(OC_ret_fut.ix[:,0],OC_ret_fut.ix[:,2],20))
plt.plot(pd.rolling_mean(CO_ret_fut.mean(axis=1)/CO_ret_fut.std(axis=1),20))
plt.plot(pd.rolling_mean(OC_ret_fut.mean(axis=1)/OC_ret_fut.std(axis=1),20))

sample = pd.concat([portCOOC_fut.mean(axis=1),pd.rolling_mean(OC_ret_fut.std(axis=1),63).shift(1),portCOOC_fut.mean(axis=1).shift(1),OC_ret_fut.std(axis=1).shift(1)],axis=1)
sample.dropna(axis=0, inplace=True)
model = sm.OLS(sample.ix[:,0],sample.ix[:,1:])
results_fu = model.fit()
print results_fu.summary()
