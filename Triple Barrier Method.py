#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Structures and ops
import pandas as pd
import numpy as np

#Viz
import matplotlib.pyplot as plt
import seaborn as sns

#Fetch
import yfinance as yf


# In[66]:


#Set the ticker
ticker = yf.Ticker("GOOG")

#Get tick data
stock = yf.download("AAPL", start = '2000-01-01', end = '2010-12-31')

price = stock['Close']


# In[76]:


#Determine the volatility approach
df0=stock.Close.index.searchsorted(stock.Close.index-pd.Timedelta(days=1))
print(pd.Series(df0).value_counts())

print('Number of sample days is {}'.format(len(stock)))

print('Duplicates: {} - {} = {}'.format(len(pd.Series(df0).value_counts()), len(stock), len(stock) - len(pd.Series(df0).value_counts())))


# In[79]:


#Volatility Measures

#Standard volatility
def getVol(stock):
    
    '''Return a volatility tracker of the stock'''
    
    stock['log_ret'] = np.log(stock['Close'] / stock['Close'].shift(1))
    
    stock['Volatility'] = stock['log_ret'].rolling(window = 252).std() * np.sqrt(252)
    
#With large duplicates and NaN
def get_Daily_Volatility(close,span0=20):
    
    '''For daily data that includes weekends'''
    
    df0=close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0=df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    
    return df0

#Average True Return
def get_atr(stock, win=14):
    
    atr_df = pd.Series(index=stock.index)
    high = pd.Series(Apple_stock.high.rolling(                      win, min_periods=win))
    low = pd.Series(Apple_stock.low.rolling(                     win, min_periods=win))
    close = pd.Series(Apple_stock.close.rolling(                       win, min_periods=win))    
          
    for i in range(len(stock.index)):
        tr=np.max([(high[i] - low[i]),                   np.abs(high[i] - close[i]),                   np.abs(low[i] - close[i])],                   axis=0)
        atr_df[i] = tr.sum() / win
     
    return  atr_df


# In[80]:


#Get the volatility
daily_volatility = get_Daily_Volatility(price)

#Days to hold the stock for the boundary width
t_final = 10

#uppoer and lower multipliers
u_l = [2,2]

#Allign the index
prices = price[daily_volatility.index]


# In[96]:


def get_3_barriers():
    
    '''Create the barrier domain'''
    
    barriers = pd.DataFrame(columns = ['days_passed', 'price','vert_barrier','top_barrier',
                                       'bottom_barrier'], index = daily_volatility.index)
    
    for day, vol in daily_volatility.iteritems():
        days_passed = len(daily_volatility.loc[daily_volatility.index[0] : day])
        
        
        #Verticla barrier
        if (days_passed + t_final < len(daily_volatility.index) and t_final != 0):
            
            vert_barrier = daily_volatility.index[days_passed + t_final]
            
        else:
            
            vert_barrier = np.nan
        
        barriers.loc[day, ['days_passed', 'price', 
        'vert_barrier']] = days_passed, prices.loc[day], vert_barrier
        
        #Top barrier
        if u_l[0] > 0:
            top_barrier = prices.loc[day] + prices.loc[day] * u_l[0] * vol
            
        else:
            top_barrier = pd.Series(index = prices.index)
            
        if u_l[1] > 0:
            bottom_barrier = prices.loc[day] - prices.loc[day] * u_l[1] * vol
            
        else:
            bottom_barrier = pd.Series(index = prices.index)
            
        barriers.loc[day, ['days_passed', 'price', 
        'vert_barrier','top_barrier', 'bottom_barrier']] = days_passed, prices.loc[day], vert_barrier, top_barrier, bottom_barrier
       
    return barriers


# In[98]:


barriers = get_3_barriers()
barriers


# In[57]:





# In[ ]:




