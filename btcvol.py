import statistics as st
import random
import numpy as np
import pandas as pd
import gdax


#Here is a function movie to create running stats

def rolling_apply(fun, a, w):
    r = np.empty(a.shape)
    r.fill(np.nan)
    for i in range(w - 1, a.shape[0]):
        r[i] = fun(a[(i-w+1):i+1])
    return r

#Main script begins here

client = gdax.PublicClient()
a = client.get_product_historic_rates('BTC-USD', granularity=60*60*24)
b = pd.DataFrame(a)
b.columns = [ 'time', 'low', 'high', 'open', 'close', 'volume' ]

c = b.iloc[:,3]
 
print  "The Overall Mean is ", st.mean(c)
print  "The Overall variance is", st.variance(c)

moving5daverage = rolling_apply(st.mean,c,5)
rolling5dvar = rolling_apply(st.variance,c,5) 

#put into dataframe
print moving5daverage
print rolling5dvar

