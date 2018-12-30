# MNISTデータのダウンロード＆読み込み
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

import tensorflow as tf
import os


# Softmax回帰の実装
# y = softmax(Wx + b)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 正解入力用のプレースホルダー
y_ = tf.placeholder(tf.float32, [None, 10])
# 交差エントロピーによる訓練
# H(y) = -Σy'log(y)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
# 最適化アルゴリズムを決定
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
# InteractiveSessionでモデルをlaunch
sess = tf.InteractiveSession()
tf.global_variables_initializer().run() # 変数の初期化演算
saver = tf.train.Saver() # Saverは変数の初期化後に生成しなければならない
for _ in range(1000): # 訓練ステップ1000回実行
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
saver.save(sess, fr'{os.getcwd()}\my-mnist-model') # モデルを保存(Windows版では絶対パスを指定すること)
'''

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.restore(sess, fr'{os.getcwd()}\my-mnist-model')
  
  # これはブール値のリストを与える
  # どのくらいの割合が正しいか決定するために、浮動小数点数値にキャストして平均を取る
  # 例えば、[True, False, True, True] は [1,0,1,1] となり、これは 0.75 となる
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  # テストデータ上で精度を求める
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # →精度はおよそ92%
