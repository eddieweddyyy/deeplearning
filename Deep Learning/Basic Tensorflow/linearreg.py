import tensorflow as tf

# height = [170, 180, 175, 160]
# shoes = [260, 270, 265, 255]

# y = ax + b

height = 170
shoes = 260

# shoes = height * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def lossfunc():
    predict = height * a + b
    return tf.square(shoes - predict)


opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    opt.minimize(lossfunc, var_list=[a,b]) #(loss func, weight Variable list)
    print(a.numpy(),b.numpy())
