# pix2pix

## Environment

- OS:
    - Windows 10
    - Ubuntu 18.04
- GPU: GeForce RTX 2060
- Python: 3.7.6 (Anaconda 4.8.2)
    - tensorflow: 1.15

### Setup
```bash
# install tensorflow-gpu 1.15
$ conda install tensorflow-gpu=1.15

# confirm tensorflow and gpu
$ python
>>> from tensorflow.python.client import device_lib
>>> device_lib.list_local_devices()
>>> exit()
```

***

## pix2pix 動作確認

```bash
# clone pix2pix git
$ git clone https://github.com/affinelayer/pix2pix-tensorflow

# download image data
$ cd pix2pix-tensorflow
$ python tools/download-dataset.py facades

# train model
## image A: 写真画像
## image B: 抽象画像
## --which_direction: 学習方向 => BtoA: 抽象画像から写真画像を生成できるように学習
$ python pix2pix.py --mode train --output_dir facades_train --max_epochs 100 --input_dir facades/train --which_direction BtoA

## => ~ 66 min (Intel Core i7-9750H 2.60GHz, GeForce RTX 2060)
## => facades_train/ に学習済みモデルが作成される
```

### Failed to get convolution algorithm. エラーが発生する場合
GeForce RTX を使っている場合、cuDNN の初期化に失敗してエラーが発生することがある

この場合、tensorflow の Session に以下の設定を渡すと上手く動く

```python
# config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# pix2pix の場合、以下の2行に config を渡す
## 616行目付近
with tf.Session(config=config) as sess:
    ...

## 713行目付近
with sv.managed_session(config=config) as sess:
    ...
```

### 学習結果の確認
```bash
# tensorboard による学習ログの確認
$ tensorboard --logdir=facades_train

## => http://localhost:6006

# 学習済みモデルを使った画像生成のテスト
## facades/val/ にあるテスト用の画像を使って検証する
$ python pix2pix.py  --mode test --output_dir facades_test --input_dir facades/val  --checkpoint facades_train

## => ~ 25 sec
## => facades_test/index.html に結果レポートが生成される
### 左: 画像生成に使った抽象画像, 真ん中: 自動生成された画像, 右: 自動生成された画像と比較するための参考画像
```
