# ref: https://deepage.net/features/numpy-rl.html
import gym
import numpy as np

'''
# Q学習
Q学習とは、状態sのときにとる行動aにどの程度の価値があるのかを示す価値関数Q(s,a)を学習させる手法
考え方としては、価値関数Q(s,a)がより高い値になる行動を選んでいけばよい
(学習を進める関係上ある程度ランダムな行動を挟んだほうがよいと言われおり、この手法をℇ-グリーディー法と呼ぶ)

Q学習の最も基本的なモデルではこの価値関数の値をテーブルを使って表す
例えば10個の状態があってそれぞれに２個ずつ行動の選択肢が存在するとすると 10x2 のテーブルでこの価値関数を表現する

今回は4つある状態変数(カートの位置・速度、ポールの角度・速度)をそれぞれ4つに区分していくため、4^4 = 256 通りの状態に分類される
この256通りの状態の中で右に押す場合、左に押す場合それぞれにおける価値を更新するため、256x2 のテーブルが必要となる

価値関数の更新は通常以下の式によってなされる
    Q(s(t), a(t)) <- (1-α)・Q(s(t), a(t)) + α・(r(t+1) + γ・max(a(t+1))・Q(s(t+1), a(t+1)))
↑
次の状態の価値観数の中で最大となる行動に減衰係数γをかけ、次の状態で得られた報酬r(t+1)を足し合わせたものをある一定の比率（学習比率）αで足し合わせることで更新

単純に報酬に直結する行動のQ値を改善するだけでは、初期の行動に対するQ値はランダムに決定された値のまま更新されないため、
次の行動に移ったとき、選択可能な行動に対するQ値の中で、最大のQ値に比例する値を直前のQ値に加える、という考えで作られたのが上記の式
こうすることで
　学習を繰り返していくと報酬を得ることができる行動そのものではなく、行動パターンに対するQ値を増加させることができる
'''

env = gym.make('CartPole-v0')

goal_average_steps = 195 # 195ステップ連続でポールが倒れないことを目指す
max_number_of_steps = 200 # 最大ステップ数
num_consecutive_iterations = 100 # 評価の範囲のエピソード数
num_episodes = 5000
last_time_steps = np.zeros(num_consecutive_iterations)

# 価値関数の値を保存するテーブルを作成（初期状態の価値関数は乱数にしておく）
## np.random.uniformは指定された範囲での一様乱数を返す
q_table = np.random.uniform(low=-1, high=1, size=(4**4, env.action_space.n)) # 256x2 のテーブル作成

# clip_min～clip_maxをnum等分した等差数列を返す（clip_minは除く）
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# ある状態sにおける4つの状態変数（カートの位置・速度、ポールの角度・速度）を離散値に変換
def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    # np.digitizeは与えられた値をbinsで指定した基数にカテゴライズする関数（カテゴリーのインデックスを返す）
    ## 今回の場合4個の離散値に変換したいため、各値の最小値～最大値を4等分した等差数列にカテゴライズすれば良い
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins = bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]
    # Q関数のテーブルは 256x2 の表であるため、カテゴライズした各値を合わせて 0~255 に変換する
    return sum([x* (4**i) for i, x in enumerate(digitized)])

# ε-グリーディー法によりQ学習を行う
'''
## ε-グリーディー法
行動選択はQ値の大きい行動を優先する
→ 初期のランダムに決まったQ値がたまたま大きな値となった行動だけが常に選択されてしまう

そこで
ある適当な定数を用意（ε = 0.3程度にすることが多い）
行動選択の際、0~1の間の乱数を生成し、その値がε以下であればランダムに行動を選択し、
εより大きければQ値の大きい行動を選択するようにすることで
Q値の初期値に依存することなく、様々な行動に対する適切なQ値の学習が可能となる
'''
def get_action(state, action, observation, reward, episode):
    next_state = digitize_state(observation)
    epsilon = 0.5 * (0.99 ** episode) # 学習が進むごとに徐々にεを小さくしていき、ランダム行動の確率を下げる
    if epsilon <= np.random.uniform(0, 1): # 乱数がε以上ならQ値の大きい行動を選択
        next_action = np.argmax(q_table[next_state])
    else: # ランダムな行動を選択
        next_action = np.random.randint([0, 1])
    # Qテーブルの更新
    # Q(s(t), a(t)) <- (1-α)・Q(s(t), a(t)) + α・(r(t+1) + γ・max(a(t+1))・Q(s(t+1), a(t+1)))
    alpha = 0.2
    gamma = 0.99
    q_table[state, action] = (1 - alpha) * q_table[state, action] + \
            alpha * (reward + gamma * q_table[next_state, next_action])
    return next_action, next_state

''' Q学習実行 '''
step_list = []
for episode in range(num_episodes):
    observation = env.reset() # 環境の初期化
    state = digitize_state(observation) # 状態を 4^4 = 256 に変換
    action = np.argmax(q_table[state]) # とりあえず初期状態のQ値の大きい行動を選択
    episode_reward = 0
    for t in range(max_number_of_steps):
        env.render() # CartPoleの描画
        observation, reward, done, info = env.step(action) # actionを取ったときの環境、報酬、状態が終わったかどうか、デバッグに有益な情報
        if done: # ゲームオーバーになった場合、報酬は-200する
            reward -= 200
        # ε-グリーディー法による行動の選択
        action, state = get_action(state, action, observation, reward, episode)
        episode_reward += reward
        if done: # ゲームオーバーになるまでのステップ数を出力
            print('%d Episode finished after %f time steps / mean %f' %
                (episode, t + 1, last_time_steps.mean()))
            # np.hstack関数は配列をつなげる関数
            last_time_steps = np.hstack((last_time_steps[1:], [t+1]))
            # 継続したステップ数をステップのリストの最後に加える
            step_list.append(t+1)
            break
    if (last_time_steps.mean() >= goal_average_steps): # 直近の100エピソードの平均が195以上であれば成功
        print('Episode %d train agent successfully!' % episode)
        break

env.close()

'''
# 学習の結果
2018/12/21 17:15～18:00
平均的に100ステップ立ち続けられるようになるまで、およそ200回試行
平均的に190ステップ立ち続けられるようになるまで、およそ900回試行
'''

# 学習曲線を表示
import matplotlib.pyplot as plt
plt.plot(np.arange(len(step_list)), step_list)
plt.xlabel('episode')
plt.ylabel('max_step')
plt.show()