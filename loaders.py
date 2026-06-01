import os
import numpy as np
import pandas as pd
import zipfile
import requests
from bs4 import BeautifulSoup
from .strategies import levret

data_folder = os.path.expanduser('~\Downloads') + '\marketdata' "\\"

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

def download_stooq(names,f="d",apikey='oHDJLRuxiTZo6zpErBdhsS7GcFeOv824'):
    stooq_path = "https://stooq.com/q/d/l/?s="  
    for i in range(0,len(names)):
        if names[i] not in ["MX_SXF","^HSCE"]:
            series = pd.read_csv(stooq_path+names[i]+"&d1=18710101&d2=20291231&apikey="+apikey+"&i="+f, header=0)
            series.to_csv(data_folder+names[i]+"_d.csv", index=False)

def load_fred(rates_names,col_names="",f="d",col=3,trans="",fgst="",fq="",start="",end=""):
    rates = []
    for i in range(0,len(rates_names)):
        if (rates_names[i]=="null"):
            tmp = local_load("null","d")
        else:
            if (rates_names[i] in ["HKD_IR","ukoeur3m","eu_prod","expplncum","impplncum","impeur"]):
                tmp = local_load(rates_names[i],f,col)
                #tmp = local_load("null","d")
            else:
                tmp = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id="+rates_names[i]+"&cosd="+start+"&coed="+end+"&fq="+fq+"&fgst="+fgst+"&transformation="+trans, index_col=[0], parse_dates=True)
            if(f!="d"):
                tmp = tmp.resample('ME').last()
        rates.append(tmp)
    rates = pd.concat(rates,axis=1, ignore_index=False)

    if (col_names==""):
        rates.columns = rates_names
    else:
        rates.columns = col_names
    rates.index.names = ['Date']
    rates = rates[rates.index>='1971-01-01']
    return rates

def local_load(name,f="d",col=3,ftype="csv"):
    if(f!=""):
        f = "_"+f
    if(ftype!="mst"):
        series = pd.read_csv(data_folder+name+f+"."+ftype, index_col=[0], parse_dates=True)
    else:
        series = pd.read_csv(data_folder+name+f+"."+ftype, index_col=[1], parse_dates=True)
        del series[series.columns[0]]
    return series.iloc[:,col]

def load_indices_loc(names,f="d",col=3,ftype="csv"):
    index = []
    newnames = []
    for n in names:
        index.append(local_load(n,f=f,col=col,ftype=ftype))
        newnames.append(n.replace('.', '_').replace('^', '_').replace('+', '_'))
    index = pd.concat(index,axis=1)
    #index = index['1971-01-01/']
    index.columns = newnames
    index.ffill(inplace=True)
    #index.reset_index(inplace=True)
    return index

def load_marketwatch(name,col='Close'):
    mypath = data_folder+name
    myfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    out=[]
    for f in myfiles:
        out.append(pd.read_csv(os.path.join(mypath, f), index_col=[0], parse_dates=True)[col])
    out = pd.concat(out,axis=0)
    out.sort_index(inplace=True)
    out.name = name.replace('.', '_')
    test = int(out.index.duplicated().sum())
    if test==0:
        return out
    else:
        return out.groupby(out.index).first()
        
def rollfutures(df_list,dir=1,cost=0,start=100):
    out = pd.concat(df_list, axis=1, sort=True)
    name = out.columns[1]
    out.columns = ['frontmonth','volcontinous']
    out.ffill(inplace=True)
    out['base'] = out.volcontinous - out.frontmonth
    out['test'] = out.volcontinous == out.frontmonth
    out['dtest'] = out.test.astype(int).diff()
    if dir==1:
        out['pnl'] = out.volcontinous.diff()
        out.loc[out['dtest']==-1,'pnl'] = out.pnl - out.base - cost
    elif dir==-1:
        out['pnl'] = -out.volcontinous.diff()
        out.loc[out['dtest']==-1,'pnl'] = out.pnl + out.base - cost
    out.loc[out.index[0],'pnl'] = start # out.volcontinous.iloc[0]
    out[name] = out.pnl.cumsum()
    return out[name]

def rollfutureslog(df_list,dir=1,cost=0,start=100,frac=0.5,danger=None):
    out = pd.concat(df_list, axis=1, sort=True)
    name = out.columns[1]
    out.columns = ['frontmonth','volcontinous']
    out.ffill(inplace=True)
    out['base'] = out.volcontinous - out.frontmonth
    out['test'] = out.volcontinous == out.frontmonth
    out['dtest'] = out.test.astype(int).diff()
    if danger is not None:
        frac = out.volcontinous.shift(1)/(danger - out.volcontinous.shift(1))    
    if dir==1:
        out['pnl'] = out.volcontinous.diff()
        out.loc[out['dtest']==-1,'pnl'] = out.pnl - out.base - cost
    elif dir==-1:
        out['pnl'] = -out.volcontinous.diff()
        out.loc[out['dtest']==-1,'pnl'] = out.pnl + out.base - cost
    out['pnl'] = np.log(1 + frac*out['pnl']/out.volcontinous.shift(1)) # logret = ln(x2/x1) = ln(1 + [x2-x1]/x1) = ln(1+dx/x1)
    out.loc[out.index[0],'pnl'] = np.log(start) # out.volcontinous.iloc[0]
    out[name] = np.exp(out.pnl.cumsum())
    return out[name]

def rollfutureslog_(df_list,dir=1,cost=0,start=100,frac=0.5,danger=None):
    out = pd.concat(df_list, axis=1, sort=True)
    name = out.columns[1]
    out.columns = ['frontmonth','volcontinous']
    out.ffill(inplace=True)
    out['base'] = np.log(out.volcontinous) - np.log(out.frontmonth)
    out['test'] = out.volcontinous == out.frontmonth
    out['dtest'] = out.test.astype(int).diff()
    if danger is not None:
        frac = out.volcontinous.shift(1)/(danger - out.volcontinous.shift(1))
    if dir==1:
        out['pnl'] = levret(frac,np.log(out.volcontinous).diff())
        out.loc[out['dtest']==-1,'pnl'] = out.pnl + levret(-frac,out.base) - levret(frac,np.log(1+cost/out.volcontinous.shift(1)))
    elif dir==-1:
        out['pnl'] = levret(-frac,np.log(out.volcontinous).diff())
        out.loc[out['dtest']==-1,'pnl'] = out.pnl + levret(frac,out.base) - levret(frac,np.log(1+cost/out.volcontinous.shift(1)))
    out.loc[out.index[0],'pnl'] = np.log(start)#out.volcontinous.iloc[0])
    out[name] = np.exp(out.pnl.cumsum())
    return out[name]
    
def get_multpl_data_manual(url):  
    # Headers to mimic a real web browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }    
    try:
        # 1. Fetch HTML and remove the estimate symbol '†'
        response = requests.get(url, headers=headers)
        clean_html = response.text.replace('†', '')

        # 2. Use the standard internal 'html.parser' (No extra installs needed)
        soup = BeautifulSoup(clean_html, 'html.parser')
        
        # 3. Manually find the table and its rows
        table = soup.find('table')
        rows = table.find_all('tr')
        
        data = []
        for row in rows:
            cols = row.find_all('td')
            # Only grab rows that have exactly two columns (Date and Value)
            if len(cols) == 2:
                date_text = cols[0].text.strip()
                value_text = cols[1].text.replace('%', '').strip()
                data.append([date_text, value_text])

        # 4. Create the DataFrame manually
        df = pd.DataFrame(data, columns=['Date', 'Value'])
        df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        return df.sort_values('Date', ascending=False)

    except Exception as e:
        print(f"Manual parsing failed: {e}")
        return None

def load_multpl(names):
    out = []
    for name in names:
        url = "https://www.multpl.com/" + name + "/table/by-month"
        df_multpl = get_multpl_data_manual(url)
        df_multpl.Date = pd.to_datetime(df_multpl.Date)
        if name=='s-p-500-dividend-yield':
            df_multpl['Date'] = df_multpl['Date'] + pd.Timedelta(days=1) #- pd.DateOffset(months=1)
            #df_multpl.iloc[1,0] = df_multpl.iloc[1,0] + pd.DateOffset(months=1) + pd.Timedelta(days=2)
        df_multpl = df_multpl.set_index('Date')
        df_multpl = df_multpl.sort_index()
        df_multpl.columns=[name.replace('-', '_')]
        out.append(df_multpl)
    return pd.concat(out,axis=1)

def correct_fred_index(df,periods=1,freq="months"):
    df['Date'] = df.index
    if freq=="months":
        df['Date'] = df['Date'] + pd.DateOffset(months=1+periods) - pd.Timedelta(days=1)
    elif freq=="weeks":
        df['Date'] = df['Date']+ pd.DateOffset(weeks=periods) - pd.Timedelta(days=1) 
    df = df.set_index('Date')
    return df