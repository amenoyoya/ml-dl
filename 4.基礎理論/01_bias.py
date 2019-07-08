# numpyを使うことで計算を高速化できる
import numpy as np

# パーセプトロン
# when: 入力信号×各重みづけの和にバイアスを加算したものが0を超える => 1
# else: => 0
def perceptron(x, w, b):
    '''
    params:
        x: numpy.array = 入力信号のコレクション
        w: numpy.array = 重みのコレクション
        b: number = バイアス
    return:
        1: if x1w1 + x2w2 + b > 0
        0: else
    '''
    # numpy.array は行列計算ができる
    if np.sum(x * w) + b > 0:
        return 1
    return 0

# AND回路
# when: x[0] = 1, x[1] = 1 => 1
# else: => 0
def AND(x):
    w = np.array([0.5, 0.5])
    b = -0.75
    return perceptron(x, w, b)

# Test: AND
def test_AND(x, y):
    '''
    params:
        x: np.array([[x1, x2], ...]) = 入力信号のペアのコレクション
        y: np.array = 期待される出力のコレクション
    return:
        true:  if test is ok
        false: else
    '''
    for i in range(y.shape[0]):
        if y[i] != AND(x[i]):
            return False
    return True

## 入力値
x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
## 期待される出力値
y = np.array([0, 0, 0, 1])
## Test
if test_AND(x, y):
    print('ANDパーセプトロン: 正常動作')
else:
    print('ANDパーセプトロン: 動作不良')
