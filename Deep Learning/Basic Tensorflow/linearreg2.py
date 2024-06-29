import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]


a = tf.Variable(0.1) #randomize
b = tf.Variable(0.1)


def lossfunc(a, b):
    prdct_y = train_x * a + b
    return tf.keras.losses.mse(train_y, prdct_y)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(2900):
    opt.minimize(lambda:lossfunc(a, b), var_list=[a,b]) #(loss func, weight Variable list)
    print(a.numpy(),b.numpy())
