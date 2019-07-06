# パーセプトロン
# when: 入力信号×各重みづけの和が閾値を超える => 1
# else: => 0
def perceptron(x, w, theta):
    '''
    params:
        x: tuple = 入力信号のコレクション
        w: tuple = 重みのコレクション
        theta: number = 閾値
    return:
        1: if x1w1 + x2w2 > theta
        0: else
    '''
    if x[0] * w[0] + x[1] * w[1] <= theta:
        return 0
    return 1

# AND回路
# when: x1 = 1, x2 = 1 => 1
# else: => 0
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.75
    return perceptron((x1, x2), (w1, w2), theta)

# Test: AND
def test_AND(x, y):
    '''
    params:
        x: tuple((x1, x2), ...) = 入力信号のペアのコレクション
        y: tuple = 期待される出力のコレクション
    return:
        true:  if test is ok
        false: else
    '''
    for i in range(len(y)):
        if y[i] != AND(x[i][0], x[i][1]):
            return False
    return True

## 入力値
x = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
)
## 期待される出力値
y = (0, 0, 0, 1)
## Test
if test_AND(x, y):
    print('ANDパーセプトロン: 正常動作')
else:
    print('ANDパーセプトロン: 動作不良')
