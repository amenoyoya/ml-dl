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
