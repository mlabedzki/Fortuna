import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
from . import strategies as st

def trimdf(obj,begin=None,end=None):
    if isinstance(obj, st.Portfolio):
        df = obj.returns
    else:
        df = obj
    if begin==None:
        begin = min(df.index)
    if end==None:
        end = max(df.index)
    if isinstance(obj, st.Portfolio):
        result = st.Portfolio(obj.weights[(df.index >= begin) & (df.index <= end)],df[(df.index >= begin) & (df.index <= end)])
    else:
        result = df[(df.index >= begin) & (df.index <= end)]
    return result

def reindex_fcst(y,newindex):
    out = []
    for x in y:
        out.append(pd.Series(x.values,index=newindex))
    return out

def reindex_fcstfull(y,newindex):   
    n = len(y[0])
    n2 = len(newindex)
    out = []
    for x in y:
        part1 = trimdf(x,None,x.index[n-n2-1])
        part2 = trimdf(x,newindex[0],None)
        part2 = pd.Series(part2.values,index=newindex)
        out.append(pd.concat([part1, part2]))
    return out

def dailyzation(df,series):
    df = series.to_frame().join(df,how='outer')
    df.ffill(inplace=True)
    del df[series.name]
    df = series.to_frame().join(df,how='left')
    del df[series.name]  
    return df   

# def dailization(df_m,df_d):
#     df = df_m.shift()
#     df = pd.DataFrame(df.values, index = df.index - dt.timedelta(days=1), columns=df.columns)
#     df = pd.merge(df_d.iloc[:,0], df, how="left").iloc[:,1:]
#     df.ffill(inplace=True)
#     return df

# dates = pd.date_range(start='2023-01-01', end='2023-03-15', freq='D')
# daily_ts = pd.DataFrame({'Value': np.random.rand(len(dates)) * 100}, index=dates)
# monthly_ts = daily_ts.resample('ME').last() # Aggregate using the mean value
# merged_df = pd.merge(left_df, right_df, left_index=True, right_index=True, how='outer')

def pltcum(obj):
    if isinstance(obj, st.Portfolio):
        x = obj.calculate_portfolio_returns()   
    else:
        x = obj
    return plt.plot(x.cumsum())  

def plot_auc_curve(fpr, tpr):
    """
    Plots the ROC AUC curve given fpr and tpr.

    Args:
        fpr (list or array): False Positive Rates.
        tpr (list or array): True Positive Rates.
    """
    roc_auc = sklm.auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
def plotmanyts(*ts_list):  
    # Combine all time series and drop rows with missing values
    df = pd.concat(ts_list, axis=1)
    df.dropna(inplace=True)
    
    # Initialize the plot layout
    plt.figure(figsize=(16, 8), dpi=150)
    
    # Dynamically plot each column using its original Series name
    for col in df.columns:
        df[col].plot(label=col)   
        
    # Set labels, legend, and grid properties
    plt.xlabel('Years')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
