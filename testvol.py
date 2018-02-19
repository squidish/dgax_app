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

n = 100
x=np.empty(n)
for i in range(1,n):
   x[i] = random.random()

print  "The Overall Mean is ", st.mean(x)
print  "The Overall variance is", st.variance(x)

moving5daverage = rolling_apply(st.mean,x,5)
rolling5dvar = rolling_apply(st.variance,x,5) 

print rolling5dvar

#put into dataframe

