'''
# Chainerによる手書き数字(MNIST)認識
8x8(=64)ピクセルで、グレースケールが17階調に設定されている手書き数字を学習し、
0～9のラベルに分類する
'''
import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from chainwrapper_v1 import run_training

'''
# ニューラルネットワーク構造
今回は畳み込みニューラルネットワーク等は使わず、
単純に全結合層/ReLU活性を3つ重ねる
'''
class Model(chainer.Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(64, 100)  # 入力64，中間層100
            self.l2 = L.Linear(100, 100) # 入力100，中間層100
            self.l3 = L.Linear(100, 10)  # 入力100，出力10
            
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

epochs = 100
batchsize = 100

# データの作成
digits = load_digits()
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target, test_size=0.2)
## GPUを使うため、numpy -> cupy に変換
data_train, data_test, label_train, label_test = cp.asarray(data_train), cp.asarray(data_test), cp.asarray(label_train), cp.asarray(label_test)
## 学習用にデータ変換
data_train = (data_train).astype(cp.float32)
data_test = (data_test).astype(cp.float32)
train = chainer.datasets.TupleDataset(data_train, label_train)
test = chainer.datasets.TupleDataset(data_test, label_test)

# ニューラルネットワークの作成
## クラス分類問題のため、損失関数にソフトマックス交差エントロピーを使用
model = L.Classifier(Model(), lossfun=F.softmax_cross_entropy)

# 学習開始
run_training(model, train, epochs, chainer.optimizers.Adam(), # 最適化関数＝Adam
    batchsize=batchsize, validation=test,
    gpu_device=0, # GPU使用
    extensions=[
        extensions.LogReport(), # ログ表示
        extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time']), # 計算状態の表示
        extensions.dump_graph('main/loss'), # ニューラルネットワークの構造
        extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='03.loss.png'), # 誤差のグラフ
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='03.accuracy.png'), # 精度のグラフ
    ]
)
# ニューラルネットワーク構造(dump_graph('main/loss'))の可視化
## > conda install -c anaconda graphviz
## > dot -Tpng result/cg.dot -o result/cg.png

'''
# 結果
20エポックあたりで精度は定常状態になっている（精度98.00%）
その後は過学習が起こっていると考えると、このニューラルネットワークでは20エポックの学習が最適と考えられる
'''