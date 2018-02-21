import statistics as st
import random
import numpy as np
import pandas as pd
import gdax
import tensorflow as tf
from random import randint
#------------------------------------------------
#Here is a function movie to create running stats
def rolling_apply(fun, a, w):
    r = np.empty(a.shape)
    r.fill(np.nan)
    for i in range(w - 1, a.shape[0]):
        r[i] = fun(a[(i-w+1):i+1])
    return r
#function for getting batches
#def next_batch(batch_size, n_steps):
def get_next_batch(x,n):
  m = randint(0,x.shape[0])
  while m > (x.shape[0]-n-1):
     m = randint(0,x.shape[0])
  return x.iloc[m:m+n,0].values,  x.iloc[m:m+10,1:].values
#-----------------------------------------------
#Main script begins here
#Prepare the data
client = gdax.PublicClient()
a = client.get_product_historic_rates('BTC-USD', granularity=60*60*24)
b = pd.DataFrame(a)
b.columns = [ 'time', 'low', 'high', 'open', 'close', 'volume' ]
c = b.iloc[:,3]
print  "The Overall Mean is ", st.mean(c)
print  "The Overall variance is", st.variance(c)
moving5daverage = rolling_apply(st.mean,c,5)
rolling5dvar = rolling_apply(st.variance,c,5) 
b['5dav'] = moving5daverage
b['5dvol'] = rolling5dvar
d = b.dropna(thresh=8)
#-----------------------------------------------
#Prepare the RNN
n_steps = 20 
n_inputs = 1
n_neurons = 100
n_outputs = 8

X = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu )
#outputs,states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu), output_size=n_outputs)
outputs,states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
#define leaning inputs
learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
#execution phase
n_iterations = 10000
batch_size = 10
#---------
#print(get_next_batch(d,10))
#
with tf.Session() as sess:
  init.run()
  for iteration in range(n_iterations):
    X_batch,y_batch = get_next_batch(d,batch_size) #fetch the next training batch
    sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
    if iteration % 100 == 0:
       mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
       print(iteration, "\tMSE:",mse)
     
