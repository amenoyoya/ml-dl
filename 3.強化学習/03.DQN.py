# ref: https://qiita.com/sugulu/items/bc7c70e6658f204f85f9
'''
# Qネットワーク
テーブルで表現していたQ関数をニューラルネットワークで近似したもの
入力層のニューロン数は、状態の次元数となる
    今回の場合、カートの位置・速度、ポールの角度・速度になるため、入力層は4つ
    Q学習ではテーブル表現するために、状態変数の離散化を行っていたが、Qネットワークでは連続値をそのまま入力する
出力層のニューロン数は、選択できる行動の数となる
    今回の場合、カートを右か左に押すだけなので、出力層は2つ

## Qネットワークの学習
Qネットワークの重みを変化させて、より良いQネットワークを実現する

状態s(t)でa(t)=右に押す　の場合、Qネットワークの出力層、右ニューロンはQ(s(t), 右）という値を出力する
    時刻tの時点で出力してほしいのは、r(t)+γ・MAX[Q(s(t+1), a(t+1))]
        ※この教師信号も本当は学習途中
        ※r(t)は時刻tでもらう報酬、γは時間割引率
    そのため、Q(s(t), 右）= r(t)+γ・MAX[Q(s(t+1), a(t+1))]
    となれば、学習は終了していることになる
この、Q(s(t), 右に押す）と r(t)+γ・MAX[Q(s(t+1), a(t+1))]の差が小さくなる方向にQネットワークの重みを更新していくことになる

## DQN特有の工夫として今回採用したもの
1. Experience Replayは学習内容をメモリに保存し、ランダムにとりだして学習する
2. Fixed Target Q-Networkは、1step分ずつ学習するのでなく、複数ステップ分をまとめて学習（バッチ学習）する
3. 報酬のclippingは、各ステップでの報酬を-1から1の間とする
    今回は各ステップで立っていたら報酬0、こけたら報酬-1、195 step以上立って終了したら報酬+1とクリップ
4. 今回の誤差関数は、誤差が1以上では二乗誤差でなく絶対値誤差を使用するHuber関数を実装

## DDQN
DDQN(Double DQN)は行動価値関数Qを、価値と行動を計算するメインのQmainと、MAX[Q(s(t+1), a(t+1))]を評価するQtargetに分ける方法
分けることで、Q関数の誤差が増大するのを防ぐことができる

今回は、試行ごとにQtargetを更新することでDDQNを実現
各試行ではQtargetはひとつ前の試行のQmainの最終値を使用
'''
import gym, time
from gym import wrappers  # gymの画像保存
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from collections import deque
import tensorflow as tf

# 損失関数の定義
## 損失関数にHuber関数を使用
### 参考: https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

# Q関数をディープラーニングのネットワーククラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]
            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            targets[i][action_b] = target               # 教師信号
            self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        # ※1： for文の外でfitすると1000回試行しても学習は完了しない。ミニバッチ法を使う意味はあるのだろうか？
        # self.model.fit(inputs, targets, epochs=1, verbose=0)

# Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)

# カートの状態に応じて、行動を決定するクラス
class Actor:
    # ε-greedy法により、ｔ＋１での行動を決定
    def get_action(self, state, episode, mainQN):
        epsilon = 0.001 + 0.9 / (1.0 + episode) # 徐々に最適行動のみをとるようにする
        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            action = np.random.choice([0, 1])  # ランダムに行動する
        return action


''' メイン処理 '''
# 初期設定
DQN_MODE = 0    # 1がDQN、0がDDQN
RENDER_MODE = 0 # 0は学習後も描画なし、1は学習終了後に描画

env = gym.make('CartPole-v0')
num_episodes = 1000  # 総試行回数
max_number_of_steps = 200  # 1試行のstep数
goal_average_reward = 195  # この報酬を超えると学習終了
num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
gamma = 0.99    # 割引係数
islearned = 0  # 学習が終わったフラグ
isrender = 0  # 描画フラグ
# ---
hidden_size = 16               # Q-networkの隠れ層のニューロンの数
learning_rate = 0.00001        # Q-networkの学習係数
memory_size = 10000            # バッファーメモリの大きさ
batch_size = 32                # Q-networkを更新するバッチの大記載

# Qネットワークとメモリ、Actorの生成
mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
# plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
memory = Memory(max_size=memory_size)
actor = Actor()

# メインルーチン
step_list = []
for episode in range(num_episodes):  # 試行数分繰り返す
    env.reset()  # cartPoleの環境初期化
    state, reward, done, _ = env.step(env.action_space.sample())  # 1step目は適当な行動をとる
    state = np.reshape(state, [1, 4])   # list型のstateを、1行4列の行列に変換
    episode_reward = 0
    
    # targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
    targetQN.model.set_weights(mainQN.model.get_weights())
    
    for t in range(max_number_of_steps + 1):  # 1試行のループ
        if (islearned == 1) and RENDER_MODE:  # 学習終了したらcartPoleを描画する
            env.render()
            time.sleep(0.1)
            print(state[0, 0])
        action = actor.get_action(state, episode, mainQN)   # 時刻tでの行動を決定する
        next_state, reward, done, info = env.step(action)   # 行動a(t)の実行による、s(t+1), R(t)を計算する
        next_state = np.reshape(next_state, [1, 4])     # list型のstateを、1行4列の行列に変換
        # 報酬を設定し、与える
        if done:
            next_state = np.zeros(state.shape)  # 次の状態s(t+1)はない
            if t < goal_average_reward:
                reward = -1  # 報酬クリッピング、報酬は1, 0, -1に固定
            else:
                reward = 1  # 立ったまま195step超えて終了時は報酬
        else:
            reward = 0  # 各ステップで立ってたら報酬追加

        episode_reward += 1 # 合計報酬を更新
        memory.add((state, action, reward, next_state))     # メモリの更新する
        state = next_state  # 状態更新
        
        # Qネットワークの重みを学習・更新
        if (memory.len() > batch_size) and not islearned:
            mainQN.replay(memory, batch_size, gamma, targetQN)

        if DQN_MODE:
            # targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
            targetQN.model.set_weights(mainQN.model.get_weights())

        # 1試行終了時の処理
        if done:
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, total_reward_vec.mean()))
            step_list.append(t+1) # 立ったままでいられたステップ数を保存
            break

    # 複数施行の平均報酬で終了を判断
    if total_reward_vec.mean() >= goal_average_reward:
        print('Episode %d train agent successfuly!' % episode)
        islearned = 1
        if isrender == 0:   # 学習済みフラグを更新
            isrender = 1

env.close()

'''
# 学習の結果
## DQN　※1 for文の外でfitした場合
2018/12/21 19:00～19:40
平均的に100ステップ立ち続けられるようになるまで、およそ900回試行

## DQN　※1 for文の中でfitした場合
2018/12/21 19:45～20:15
平均的に100ステップ立ち続けられるようになるまで17回試行
平均的に190ステップ立ち続けられるようになるまで98回試行

## DDQN　※1 for文の中でfitした場合
2018/12/21 20:15～20:25
平均的に100ステップ立ち続けられるようになるまで16回試行
平均的に190ステップ立ち続けられるようになるまで29回試行
'''

# 学習曲線を表示
import matplotlib.pyplot as plt
plt.plot(np.arange(len(step_list)), step_list)
plt.xlabel('episode')
plt.ylabel('max_step')
plt.show()