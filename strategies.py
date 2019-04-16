import pandas as pd
import numpy as np
import loaders as ld
import analytics as an

def cooc(opens,closes,costs=0.0001,cutoff=1,strength=10000,prec=2,ret=0,sw=0,inv=0):
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
