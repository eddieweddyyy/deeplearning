import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv')

# print(data.isnull().sum())
data = data.dropna()
# data = data.fillna(100)
ydata = data['admit'].values
# xdata = data.drop('admit', axis=1).values

xdata = []
for i, rows in data.iterrows():
  xdata.append([rows['gre'], rows['gpa'], rows['rank']]) 
# print(xdata)

# xdata = tf.convert_to_tensor(xdata)
# ydata = tf.convert_to_tensor(ydata)

xdata = np.array(xdata)
ydata = np.array(ydata)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid') #sigmoid => range 0 - 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #binary_crossentropy => range 0 - 1
model.fit(xdata, ydata, epochs=1000) #model.fit(input, result, epochs=int)

prdct = model.predict([[750], [3.70], 3], [400, 2.2, 1], [900, 4.5, 1])
print(prdct)
