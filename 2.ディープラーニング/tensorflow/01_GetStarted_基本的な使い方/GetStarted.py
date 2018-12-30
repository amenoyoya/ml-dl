import tensorflow as tf

# 2×3行列を生成するconstant（定数）OP（演算）を作成
matrix1 = tf.constant([
  [1., 2., 3.],
  [4., 5., 6.]
])

# 3×2行列を生成するconstant（定数）OP（演算）を作成
matrix2 = tf.constant([
  [7., 8.],
  [9., 0.],
  [1., 2.]
])

#‘matrix1’と‘matrix2’を入力として取る Matmul OP を作成
# 戻り値‘product’は行列の乗算の結果となる
product = tf.matmul(matrix1, matrix2)

# デフォルト・グラフを実行するためのセッションを作成
with tf.Session() as sess: # withブロックによりセッションは自動的に close される
  # 明示的にGPUやCPUの使用を指定する場合、tf.device を生成
  # "/cpu:0": 貴方のマシンの CPU
  # "/gpu:0": 貴方のマシンの GPU
  # "/gpu:1": 貴方のマシンの２つ目の GPU, etc.
  # （明示的に指定しない場合、最初のGPUをできる限り多くの処理で使用）
  with tf.device("/gpu:0"):
    # matmal OP を動作させるために、matmal OP の出力を表す‘product’を渡して sessioin‘run()’メソッドを呼び出す
    result = sess.run([product])
    print(result) # => [[28, 14], [79, 44]]
