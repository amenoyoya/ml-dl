'''
# Chainerによる畳み込みニューラルネットワークで手書き数字(MNIST)認識
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
2次元畳み込みニューラルネットワークを使用

## 畳み込み
Convolution2D(in_channel, out_channel, filter_size, stride, padding)
@in_channel: 入力層の厚み。白黒画像なら1だが、RGB別にデータを用意しているなら3
@out_channel: 入力画像を何枚の画像に畳み込むか
@filter_size: 畳み込みのフィルターサイズ
              3を指定すれば3x3でフィルタリング
              (3,4)を指定すれば3x4でフィルタリング
@stride: フィルターをかけるときのずらしの大きさ（ストライド）
         32x32の画像に対しストライド1で3x3のフィルターをかけると、出力は30x30に縮小される（フィルターの左上が画像の1,2,3,...30(ここでフィルターの右上が32に達する)をたどる）
         32x32の画像に対しストライド2で3x3のフィルターをかけると、出力は15x15に縮小される（フィルターの左上が画像の1,3,5,...29(ここでフィルターを動かせなくなる)をたどる）
@padding: 畳み込みフィルターをかけると画像が縮小するため、縮小した分をゼロパディングして元の大きさに戻すために使用
          32x32の画像に対しストライド1で3x3のフィルターをかけると、出力は30x30になるが、周り1列を0で埋めれば(padding=1)、32x32の画像で出力できる

例：8x8の画像をConv2D(1,4,3)で畳み込みすると、8x8(6x6の画像の周り1列が0埋めされている)画像4枚が出力される

## プーリング
畳み込みをすると画像が増加していき計算が複雑になっていくため、プーリングにより画像を縮小し、より特徴的なパターンを際立たせる処理をする

### 最大値プーリング
max_pooling_2d(activation, filter_w, filter_h)
@activation: 活性化関数（ReLUを使うことが多い）
@filter_w, filter_h: プーリングフィルタサイズ

元画像に対してフィルターをかけ、画像分割をする。この時のストライドはフィルターサイズを同じ大きさ
例えば、8x8の画像に対して2x2のフィルターをかけると、2x2の画像が4x4枚生成する
その後、2x2の画像内の最大値のみを集め、最終的に4x4の画像として出力する

## 今回のネットワーク構造
1x(8x8)の画像入力 -> 畳み込み(1,16,3) -> 16x(8x8(6x6画像周り1列をゼロパディング)) -> 最大値プーリング(ReLU,2,2) -> 16x(4x4)
-> 畳み込み(16,64,3) -> 64x(4x4(2x2画像周り1列をゼロパディング)) -> 最大値プーリングプーリング(ReLU,2,2) -> 64x(2x2)
-> 64x2x2=256の信号を全結合層で処理 -> [0～9]にラベリング
'''
class Model(chainer.Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(1, 16, 3, 1, 1)  # 1x(8x8)の画像を16x(8x8)に畳み込み -> 最大値プーリングにより16x(4x4)に縮小
            self.l2 = L.Convolution2D(16, 64, 3, 1, 1) # 16x(4x4)の画像を64x(4x4)に畳み込み -> 最大値プーリングにより64x(2x2)に縮小
            self.l3 = L.Linear(256, 10)  # 入力64x2x2，出力10
            
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.l1(x)), 2, 2)
        h2 = F.max_pooling_2d(F.relu(self.l2(h1)), 2, 2)
        return self.l3(h2)

epochs = 100
batchsize = 100

# データの作成
digits = load_digits()
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target, test_size=0.2)
## GPUを使うため、numpy -> cupy に変換
data_train, data_test, label_train, label_test = cp.asarray(data_train), cp.asarray(data_test), cp.asarray(label_train), cp.asarray(label_test)
## 畳み込みネットワーク入力用に 1x8x8 の大きさに変換
data_train = data_train.reshape((len(data_train), 1, 8, 8))
data_test = data_test.reshape((len(data_test), 1, 8, 8))
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
        extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='04.loss.png'), # 誤差のグラフ
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='04.accuracy.png'), # 精度のグラフ
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