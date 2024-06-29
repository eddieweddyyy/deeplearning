import tensorflow as tf

t = tf.constant( [3,4,5] )
t2 = tf.constant( [6, 7, 8] )
print(t + t2)

t3 = tf.constant([[1,2],
                  [3,4]])

tf.add(t, t2)
tf.subtract(t, t2)
tf.divide(t, t2)
tf.multiply(t, t2)
tf.matmul(t, t2) #dot product
t4 = tf.zeros( [2, 2, 3])

print(t4)
print(t.shape)
tf.cast() #change the type of tensor
w = tf.Variable(1.0) #weight
print(w.numpy())
w.assign(2)