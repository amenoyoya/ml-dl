# ref: https://www.tcom242242.net/entry/2017/09/05/061405
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make('CartPole-v0')

state_shape = env.observation_space.shape # 状態変数4つ（カートの位置・速度、ポールの角度・速度）
action_size = env.action_space.n          # 行動選択肢2つ（左 or 右）
hidden_size = 16                          # 隠れ層

# モデルの定義
model = Sequential()
model.add(Flatten(input_shape=(1,) + state_shape))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(action_size, activation='linear'))
optimizer = Adam(lr=0.001)  # 誤差を減らす学習方法はAdam
print(model.summary()) # モデルの定義状態を確認

# 強化学習エージェントの定義
memory = SequentialMemory(limit=50000, window_length=1) # experience reply で用いるmemory
policy = BoltzmannQPolicy() # 行動選択手法の定義
dqn = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=10,
               target_model_update=0.01, policy=policy)
dqn.compile(optimizer, metrics=['mae'])

# コールバック関数
## EarlyStopping: episode_reward(報酬)をモニターして、定常状態になってから8エポック後に終了
early_stopping = EarlyStopping(monitor='episode_reward', mode='auto', patience=8)
## ModelCheckpoint: 1エポック学習の度にモデルを保存／DQNAgentはsave_weightsしか使えないため、save_weights_only=Trueにする
model_checkpoint = ModelCheckpoint(filepath="04.dqn_weights.h5", save_weights_only=True)

# 強化学習実行
history = dqn.fit(
    env,
    nb_steps=50000, # 1ゲームの実行に必要なstep数×学習回数(エポック数)
    visualize=True,
    verbose=2,
    callbacks=[early_stopping, model_checkpoint]
)

'''
2018/12/21 23:25～23:40
200step立ち続けられるようになるまで、83エポック
定常状態になるまで331エポック
'''

# 学習の結果を実行
dqn.test(env, nb_episodes=5, visualize=True) # 5回中4回が200ステップ耐久、1回が198ステップで、非常に高性能

# 学習曲線をプロット
import matplotlib.pyplot as plt
step_list = history.history['nb_episode_steps']
plt.plot(np.arange(len(step_list)), step_list)
plt.xlabel('episode')
plt.ylabel('max_step')
plt.show()