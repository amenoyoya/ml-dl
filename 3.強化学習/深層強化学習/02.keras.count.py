import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

# データの作成
trainX = np.array(([0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]), dtype=np.float32) # 特徴量はfloat32型にする
trainY = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int32) # 目的変数＝正答ラベル
trainY = to_categorical(trainY)

# モデル構築
model = models.Sequential()
model.add(layers.Dense(6, activation='relu', input_shape=(3,))) # 入力3、中間層6／ReLU活性化関数
model.add(layers.Dense(6, activation='relu')) # 入力6、中間層6／ReLU活性化関数
model.add(layers.Dense(4, activation='softmax')) # 入力6、出力4
# モデルのコンパイル
## 多クラス分類問題のため、損失関数にcategorical_crossentropyを採用
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) # 最適関数＝Adam

# EarlyStopping: val_accをモニターして、定常状態になってから100エポック後に終了
early_stopping = EarlyStopping(monitor='val_acc', mode='auto', patience=100)
# ModelCheckpoint: 1エポック学習の度にモデルを保存
model_checkpoint = ModelCheckpoint(filepath="02.keras.weights.h5", save_weights_only=True)

# 訓練実行
model.fit(trainX, trainY, batch_size=8, epochs=10000, shuffle=True,
    validation_split=0.2, # 訓練データのうち20％を検証データとして使用
    callbacks=[early_stopping, model_checkpoint]
)