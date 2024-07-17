text = open(r'C:\Users\CKIRUser\Desktop\eddie\coding_eddie2\Deep Learning\Project 3,4 - Sequence Learning & RNN\Composer AI\pianoabc.txt', 'r').read()
# print(text)

uniq_text = list(set(text))
uniq_text.sort()
# print(uniq_text)

# utilities
text_to_num = {}
num_to_text = {}

for i, data in enumerate(uniq_text):
  text_to_num[data] = i
  num_to_text[i] = data

# print(text_to_num)

numeratetxt = []
for i in text:
  numeratetxt.append(text_to_num[i])

print(numeratetxt)

trainX = []
trainY = []
for i in range(0, len(numeratetxt) - 25):
  trainX.append(numeratetxt[i : i+25]) #[inclusive : exclusive]
  trainY.append(numeratetxt[i+25])
# for i in range(0, len(numeratetxt) - 5, 5):
  # trainX.append([numeratetxt[i], numeratetxt[i+1], numeratetxt[i+2], numeratetxt[i+3], numeratetxt[i+4]])
  # trainY.append(numeratetxt[i+5])


print(trainX[0 : 5])
print(trainY[0 : 5])

import numpy as np

print(np.array(trainX).shape)
print(np.array(trainY).shape)


import tensorflow as tf

trainX = tf.one_hot(trainX, 31)
trainY = tf.one_hot(trainY, 31)
print(trainX[0:2])

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25,31)),
    tf.keras.layers.Dense(31, activation='softmax') #softmax & categorical_crossentropy come together

])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=64, epochs=1000, verbose=2)

model.save(r'C:\Users\CKIRUser\Desktop\eddie\coding_eddie2\Deep Learning\Project 3,4 - Sequence Learning & RNN\Composer AI\model1')