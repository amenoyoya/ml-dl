import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

# MNISTデータセット読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 畳み込みニューラルネットワーク
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 無駄な学習（過学習）を防ぐためにEarlyStoppingを設定
## 検証データの損失関数が改善されなくなってから 10エポック後に終了
stopper = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

# 28 x 28の画像がgrayscaleで1chなので、28, 28, 1にreshapeする
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 0-255の整数値を0〜1の小数に変換する
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# ラベルを one-hot vector形式に変換する
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# モデルの訓練
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=1000,
                    verbose=1,
                    validation_split=0.1, # 訓練データの1割を検証データとして利用
                    shuffle=True,
                    callbacks=[stopper] # EarlyStoppingを設定
                    )
# 学習済みモデルの評価
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])