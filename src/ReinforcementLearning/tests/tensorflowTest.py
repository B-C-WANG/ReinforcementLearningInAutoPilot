# coding:utf-8
# Type: Private Author: Baochuan Wang

'''
测试tensorflow的feeddict是否有问题
'''

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=(None, 1))
y = tf.placeholder(tf.float32, shape=(None, 1))

z = x + y

sess = tf.Session()

z = sess.run(z, feed_dict={
    x: np.random.random((10, 1)),
    y: np.random.random((10, 1))})

print(z)


# 上面这一段正常运行


# 第二个测试

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# the model of the fully-connected network
weights = tf.Variable(tf.random_normal([784, 10]))
biases = tf.Variable(tf.zeros([1, 10]) + 0.1)
outputs = tf.matmul(xs, weights) + biases
predictions = tf.nn.softmax(outputs)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predictions),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
sess.run(train_step,feed_dict={
    xs:np.random.random((1,784)),
    ys:np.random.random((1,10))
})
exit()



input_state = tf.placeholder(tf.float32, shape=(None,4))

b1 = tf.Variable(tf.zeros((1,4))+0.1,name="1")

output = input_state + b1


y = sess.run(output,feed_dict={
    input_state:np.random.random((10,4))
})
print(y)
exit()
t = input_state
for size in (16,8):
    t = tf.keras.layers.Dense(units=size, activation=tf.keras.activations.relu)(t)
output_pred_y = tf.keras.layers.Dense(units=4)(t)

output_pred_y_test =tf.keras.layers.Dense(units=4)(input_state)

output_pred_y_non_keras = tf.layers.dense(inputs=input_state,units=4)


w1 = tf.Variable(tf.random_normal(shape=(4,10)))
b1 = tf.Variable(tf.zeros(shape=(1,10))+0.1)

y = tf.matmul(input_state,w1)+ b1


y = sess.run(y,feed_dict={
    input_state :np.random.random((10,4))
})
