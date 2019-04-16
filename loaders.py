import pandas as pd
import zipfile
import quandl

def histdata(name,hour=8,begin=2010):
    out = pd.DataFrame()
    for i in range(begin,2018):
        zf = zipfile.ZipFile('HISTDATA_COM_ASCII_'+name+'_M1'+str(i)+'.zip')
        df = pd.read_csv(zf.open('DAT_ASCII_'+name+'_M1_'+str(i)+'.csv'), delimiter=';', index_col=[0], parse_dates=True, header=None)
        h = df.asfreq('1Min', how='start', method = 'ffill')
        h = h.asfreq('60Min', how='start')
        #h = pd.concat([h,pd.Series(h.index.hour, index=h.index),pd.Series(h.index.minute, index=h.index)], axis=1)
        h = pd.concat([h,pd.Series(h.index.hour, index=h.index)], axis=1)
        h.columns = range(0,6)
        #h = h.loc[h[6] == 0]
        out = out.append(h.loc[h[5] == hour])
    for i in range(1,8):
        zf = zipfile.ZipFile('HISTDATA_COM_ASCII_'+name+'_M120180'+str(i)+'.zip')
        df = pd.read_csv(zf.open('DAT_ASCII_'+name+'_M1_20180'+str(i)+'.csv'), delimiter=';', index_col=[0], parse_dates=True, header=None)
        h = df.asfreq('1Min', how='start', method = 'ffill')
        h = h.asfreq('60Min', how='start')
        h = pd.concat([h,pd.Series(h.index.hour, index=h.index)], axis=1)
        h.columns = range(0,6)
        out = out.append(h.loc[h[5] == hour])
    out = out[out.index.dayofweek < 5]
    out.index = out.index.date
    return out

def bossa_load(col,name,f="d"):
    #series = pd.read_csv(name+"_"+f+".csv", delimiter=',', index_col=[0], parse_dates=True)
    series = pd.read_csv(name+".mst", delimiter=',', index_col=[1], parse_dates=True)
    #series = series.ix[:,[0,3]]
    #column_indices = [0,1,2,3]
    column_indices = [0,1,2,3,4,5]
    #new_names = [name+'o', name+'h', name+'l', name+'c']    
    new_names = [name, name, name, name, name, name]    
    old_names = series.columns[column_indices]
    series.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    series = series.iloc[:,col]   
    return series

def xtb_load(col,name,f="d"):
    #series = pd.read_csv(name+"_"+f+".csv", delimiter=',', index_col=[0], parse_dates=True)
    series = pd.read_csv(name+".csv", delimiter=',', index_col=[0], parse_dates=True)
    #series = series.ix[:,[0,3]]
    #column_indices = [0,1,2,3]
    column_indices = [0,1,2,3,4,5]
    #new_names = [name+'o', name+'h', name+'l', name+'c']    
    new_names = [name, name, name, name, name, name]    
    old_names = series.columns[column_indices]
    series.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    series = series.iloc[:,col]   
    return series
    
def local_load(col,name,f="d"):
    series = pd.read_csv(name+"_"+f+".csv", delimiter=',', index_col=[0], parse_dates=True)
    #series = series.ix[:,[0,3]]
    column_indices = [0,1,2,3]
    #new_names = [name+'o', name+'h', name+'l', name+'c']    
    new_names = [name, name, name, name]    
    old_names = series.columns[column_indices]
    series.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    series = series.iloc[:,col]   
    return series
    
def stacklisted(names,f="d",k=0):
    listdf = []
    for i in range(0,len(names)):
        listdf.append(local_load(k,names[i],f))
    return pd.concat(listdf, axis=1)
    
def bstacklisted(names,f="d",k=0):
    listdf = []
    for i in range(0,len(names)):
        listdf.append(bossa_load(k,names[i],f))
    return pd.concat(listdf, axis=1)

def xstacklisted(names,f="d",k=0):
    listdf = []
    for i in range(0,len(names)):
        listdf.append(xtb_load(k,names[i],f))
    return pd.concat(listdf, axis=1)

def qstacklisted(name,m,k=0,s="1900-01-01"):
    listdf = []
    for i in range(3,m+1):
        tmp = quandl.get(name+str(i), start_date=s, authtoken="GU-B7tBC6aihybbeuJBX").iloc[:,k]
        listdf.append(tmp)
    out = pd.concat(listdf, axis=1)
    out.columns = range(3,m+1)
    return out

def stackhistdata(names,h,k=0,begin=2010):
    listdf = []
    for i in range(0,len(names)):
        tmp = histdata(names[i],h,begin)[k]
        listdf.append(tmp)
    out = pd.concat(listdf, axis=1)
    out.columns = names
    return out

def download_quandl(names,k=1):
    for i in range(0,len(names)):
        series = pd.read_csv('https://www.quandl.com/api/v3/datasets/CHRIS/'+names[i]+str(k)+'.csv?api_key=GU-B7tBC6aihybbeuJBX',index_col=[0])
        series.to_csv(names[i]+str(k)+".csv")

def load_quandl(name,n=1,m=3,k=0):
    listdf = []
    for i in range(n,m+1):
        tmp = pd.read_csv(name+str(i)+".csv",index_col=[0],parse_dates=True).iloc[:,k+1]
        listdf.append(tmp)
    out = pd.concat(listdf, axis=1)
    out.columns = range(n,m+1)
    out.dropna(axis=0,how='any',inplace=True)
    return out
