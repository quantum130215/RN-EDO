#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math 


# In[2]:


class ODEsolve(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss") #loss_tracker pude ser cambiada por mivalor_de_costo o como queramos
        
        
    @property
    def metrics(self):
        return [self.loss_tracker] #igual cambia el loss_tracker
    
    
    def train_step(self, data):
        batch_size = 1000 #Calibra la resolucion de la ec.dif
        x = tf.random.uniform((batch_size,1), minval=-1, maxval=1)
        
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            eq = y_pred-(3.*tf.sin(math.pi*x))
            loss = keras.losses.mean_squared_error(0.,eq)
        
        #aplica los gradientes        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        
        #actualiza metricas
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}


# In[3]:


model = ODEsolve()

model.add(Dense(10,activation ='tanh', input_shape=(1,)))
model.add(Dense(200,activation ='tanh'))
model.add(Dense(100,activation ='tanh'))
model.add(Dense(100,activation ='tanh'))
model.add(Dense(100,activation ='tanh'))
model.add(Dense(50,activation ='tanh'))
model.add(Dense(1, activation ='linear'))

model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])

x=tf.linspace(-1,1,100)
history = model.fit(x,epochs=1000,verbose=1)

x_testv = tf.linspace(-1,1,100)
a=model.predict(x_testv)


#model.save("RNA_EDO1.h5")


# In[4]:


plt.figure(figsize=(10,6))
plt.plot(x_testv,a)
plt.plot(x_testv, 3.*np.sin(np.pi*x) )
plt.show()


# In[ ]:




