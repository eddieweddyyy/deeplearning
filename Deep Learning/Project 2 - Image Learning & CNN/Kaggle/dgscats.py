# Need kaggle.json & dogs-vscats-redux-kernels-edition/test and train unzipped.
import os

os.mkdir('/content/dataset')
os.mkdir('/content/dataset/cats')
os.mkdir('/content/dataset/dogs')

import tensorflow as tf
import shutil

print(len(os.listdir('/content/train/')))

for i in os.listdir('/content/train/'):
  if 'cat' in i:
    shutil.copyfile('/content/train/' + i, '/content/dataset/cats/' + i)
  else:
    shutil.copyfile('/content/train/' + i, '/content/dataset/dogs/' + i)

# tf.keras.preprocessing.image_dataset_from_directory('')

train_ds  = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/dataset',
    image_size=(150,150),
    batch_size=64,
    subset='training',
    validation_split=0.2,
    seed=1234
)

val_ds  = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/dataset',
    image_size=(150,150),
    batch_size=64,
    subset='validation',
    validation_split=0.2,
    seed=1234
)

print(train_ds)
# Data Preprocessing-------------------
def preprocessing(i, answer):
  i = tf.cast(i/255.0, tf.float32)
  return i, answer

train_ds = train_ds.map(preprocessing)
val_ds = val_ds.map(preprocessing)
# -------------------------------------
# import matplotlib.pyplot as plt

for i, answer in train_ds.take(1):
  print(i)
  print(answer)
#   plt.imshow(i[0].numpy().astype('uint8'))
#   plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5)

from tensorflow.keras.applications.inception_v3 import InceptionV3

inception_model = InceptionV3(input_shape=(150,150,3), include_top=False)
inception_model.load_weights('inception_v3.h5')

# inception_model.summary()

for i in inception_model.layers:
  i.trainable = False

unfreeze = False
for i in inception_model.layers:
  if i.name == 'mixed6':
    unfreeze = True
  if unfreeze == True:
    i.trainable = True

last_layer = inception_model.get_layer('mixed7')

# print(last_layer)
# print(last_layer.output)
# print(last_layer.output_shape)

layer1 = tf.keras.layers.Flatten()(last_layer.output)
layer2 = tf.keras.layers.Dense(1024, activation='relu')(layer1)
drop1 = tf.keras.layers.Dropout(0.2)(layer2)
output = tf.keras.layers.Dense(1, activation='sigmoid')(drop1)

model = tf.keras.Model(inception_model.input, output)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.00001), metrics=['acc'])

model.fit(train_ds, validation_data=val_ds, epochs=2)