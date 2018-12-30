'''
Chainer用ラッパーライブラリ
'''
import chainer
from chainer import training
from chainer.training import extensions as ext

# Chainerモデルの訓練実行
## model: ニューラルネットワークモデル
## train: 訓練用データセット
## epochs: 学習回数
## optimizer: 最適関数
## batchsize: ミニバッチサイズ
## validation: 検証用データセット
## extensions: 訓練中のログ表示設定等
## gpu_device: GPUを使う場合は0（GPUが2つ以上ある場合は0以上）を指定
def run_training(model, train, epochs, optimizer, batchsize=1, validation=None, extensions=[], gpu_device=-1):
    optimizer.setup(model)
    if gpu_device >= 0: # ChainerをGPU対応に
        chainer.cuda.get_device(gpu_device).use()
        model.to_gpu(gpu_device)
    # イテレータの定義
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    # アップデータの登録
    updater = training.StandardUpdater(train_iter, optimizer)
    # トレーナーの登録
    trainer = training.Trainer(updater, (epochs, 'epoch'))
    if validation is not None: # 検証データの表示
        trainer.extend(ext.Evaluator(chainer.iterators.SerialIterator(validation, batchsize, repeat=False, shuffle=False), model))
    for extension in extensions:
        trainer.extend(extension)
    # 訓練実行
    return trainer.run()