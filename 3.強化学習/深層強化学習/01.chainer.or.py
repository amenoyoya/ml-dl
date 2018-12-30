'''
# ChainerによるニューラルネットワークでOR演算を学習

## OR演算
0か1の2値をとる2つの変数があったとして、
    いずれかが1であれば1を出力
    いずれも0であれば0を出力

## 教師あり学習
特徴量：0か1の2値をとる2変数を組み合わたデータ（＝4通り）
目的変数：OR演算の結果（0か1）を出力
'''
import cupy as np # GPU利用時はnumpyの代わりにcupyを使用
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainwrapper_v1 import run_training

epochs = 100 # 学習回数
batchsize = 4 # ミニバッチサイズ

'''
# ネットワーク構造
Input(2): 2つの変数を受け取るため、入力2つ
-> 全結合層(in=2, hidden=3)/ReLU活性化関数: 入力2つ、中間層3つ
-> 全結合層(in=3, out=2): 前の層から3つの入力が来て、出力は2つ（0 or 1 のラベル）

## ReLU活性化関数
0以下なら0に、それ以外ならそのままの数値に変換

## Kerasの場合
model.add(Dense(3, input_dim=2, activation='relu'))
     .add(Dense(2))
'''
class Model(chainer.Chain):
    def __init__(self): # ネットワーク層の定義
        super(Model, self).__init__()
        with self.init_scope():
            # なぜかネットワーク層を配列で定義すると上手く学習してくれない
            '''
            self.layers = [
                L.Linear(2, 3), # 入力2、中間層3の全結合層
                L.Linear(3, 2), # 入力3、出力2の全結合層（出力層）
            ]
            '''
            self.l1 = L.Linear(2, 3) # 入力2、中間層3の全結合層
            self.l2 = L.Linear(3, 2) # 入力3、出力2の全結合層（出力層）
    
    def __call__(self, x): # ネットワーク層間の信号の伝播方法（活性化関数）の定義
        h1 = F.relu(self.l1(x)) # 1層目の活性化関数はReLU
        return self.l2(h1) # 1層目からReLU関数を通して伝播されてきた信号を2層目が処理して出力

# データの作成
trainX = np.array(([0,0], [0,1], [1,0], [1,1], [1,0], [1,1], [0,0], [0,1]), dtype=np.float32) # 特徴量はfloat32型にする
trainY = np.array([0, 1, 1, 1, 1, 1, 0, 1], dtype=np.int32) # 目的変数＝正答ラベル
testX = np.array(([1,0], [0,1], [1,1], [0,0], [1,0], [0,1]), dtype=np.float32)
testY = np.array([1, 1, 1, 0, 1, 1], dtype=np.int32)

train = chainer.datasets.TupleDataset(trainX, trainY)
test = chainer.datasets.TupleDataset(testX, testY)

# ニューラルネットワークの作成
## 2クラス分類問題のため、損失関数にソフトマックス交差エントロピーを使用
model = L.Classifier(Model(), lossfun=F.softmax_cross_entropy)

# 学習開始
run_training(model, train, epochs, chainer.optimizers.Adam(), # 最適化関数＝Adam
    batchsize=batchsize, validation=test,
    gpu_device=0, # GPU使用
    extensions=[
        extensions.LogReport(), # ログ表示
        extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time']), # 計算状態の表示
        extensions.dump_graph('main/loss'), # ニューラルネットワークの構造
        extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='01.loss.png'), # 誤差のグラフ
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='01.accuracy.png'), # 精度のグラフ
    ]
)
# ニューラルネットワーク構造(dump_graph('main/loss'))の可視化
## > conda install -c anaconda graphviz
## > dot -Tpng result/cg.dot -o result/cg.png