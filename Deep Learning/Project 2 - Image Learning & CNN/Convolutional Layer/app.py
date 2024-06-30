import tensorflow as tf
import matplotlib.pyplot as plt



(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# print(trainX[0])
# print(trainX.shape)

# print(trainY)

plt.imshow(trainX[1])
plt.gray()
plt.colorbar()
plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

