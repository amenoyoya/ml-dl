'''
y = x * 0.1 + 0.3 となる、ダミーのデータポイント x_data, y_data を NumPy で作成し、
y_data = W * x_data + b となる W と b の適正値を TensorFlow に見つけさせる
'''
import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype("float32") # 0～1の乱数配列を100行生成し、float32にキャスト
y_data = x_data * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(201):
  sess.run(train)
  if step % 20 == 0:
    print(step, sess.run(W), sess.run(b))

sess.close()