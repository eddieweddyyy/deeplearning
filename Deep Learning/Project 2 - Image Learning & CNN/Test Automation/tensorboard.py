#Tensorboard and EarlyStopping

import tensorflow as tf
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28,28,1) )
testX = testX.reshape( (testX.shape[0], 28,28,1) )

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

import time

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format('firstmodel' + str(int(time.time()))))

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=[tensorboard])

#------------------------------------------------------------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), padding="same", activation="relu", input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])


#Save and Checkpoints----------------------------------------
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint/mnist',
    # monitor='val_acc',
    # mode='max',
    # filepath='checkpoint/mnist{epoch}',
    save_weights_only=True,
    save_freq='epoch'
)
model.save('new/model1')
loaded_model = tf.keras.models.load_model('new/model1')

loaded_model.summary()
loaded_model.evaluate(testX, testY)
# -----------------------------------------------------------
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format('Conv2model' + str(int(time.time()))))

#EarlyStopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')


model.fit(trainX, trainY, validation_data=(testX, testY), epochs=300, callbacks=[tensorboard, es]) #earlystopping
# -----------------------------------------------------------------------------------------------------------
def createModel(): #modify so that the function creates model. Use parameters
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), padding="same", activation="relu", input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
  ])
  return model
