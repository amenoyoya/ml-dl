'''
# Chainerによるニューラルネットワークで1の個数を学習

## 教師あり学習
特徴量：0か1の2値をとる3変数を組み合わたデータ（＝8通り）
目的変数：1の個数（0～3）を出力
'''
import cupy as np # GPU利用時はnumpyの代わりにcupyを使用
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainwrapper_v1 import run_training

epochs = 100 # 学習回数
batchsize = 8 # ミニバッチサイズ

class Model(chainer.Chain):
    def __init__(self): # ネットワーク層の定義
        super(Model, self).__init__()
        with self.init_scope():
            # ネットワーク層を配列で定義すると上手く学習できない
            '''
            self.layers = [
                L.Linear(3, 6), # 入力3，中間層6
                L.Linear(6, 6), # 入力6，中間層6
                L.Linear(6, 4), # 入力6，出力4
            ]
            '''
            self.l1 = L.Linear(3, 6) # 入力3，中間層6
            self.l2 = L.Linear(6, 6) # 入力6，中間層6
            self.l3 = L.Linear(6, 4) # 入力6，出力4
    
    def __call__(self, x): # ネットワーク層間の信号の伝播方法（活性化関数）の定義
        a1 = F.relu(self.l1(x))
        a2 = F.relu(self.l2(a1))
        return self.l3(a2)

# データの作成
trainX = np.array(([0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]), dtype=np.float32) # 特徴量はfloat32型にする
trainY = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int32) # 目的変数＝正答ラベル

train = chainer.datasets.TupleDataset(trainX, trainY)
test = chainer.datasets.TupleDataset(trainX, trainY)

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
        extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='02.loss.png'), # 誤差のグラフ
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='02.accuracy.png'), # 精度のグラフ
    ]
)
# ニューラルネットワーク構造(dump_graph('main/loss'))の可視化
## > conda install -c anaconda graphviz
## > dot -Tpng result/cg.dot -o result/cg.png