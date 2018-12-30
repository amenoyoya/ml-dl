'''
# Qラーニングで倒立振子問題を解く

## 倒立振子
台車（カート）の上に連結された棒（ポール）を倒さないように台車を動かす
・状態は以下の4つ
    ・カートの位置：-inf～inf
    ・カートの速度：-inf～inf
    ・ポールの角度：-180°～180°（π=-1～1）
    ・ポールの速度：-inf～inf
・行動選択肢は以下の2つ
    ・カートを左に押す
    ・カートを右に押す
・終了条件
    ・カート位置が-2.4～2.4の範囲外に行ってしまったら終了
    ・ポールが-41.8°～41.8°の範囲より外側に倒れてしまったら終了
・報酬
    ・カートを何ステップ動かし続けられたかをそのまま報酬とする
    ・上限を200とする（200ステップ以上は実行しない）

## 倒立振子のQテーブル
状態の数値は連続値であるため、それらをそのままQテーブル化することはできない
そのため数値を離散化する必要がある
今回は、カートの位置と速度、ポールの角度と速度に動作の制限を与え、それらを6分割することとした
    ・カート位置：[-2.4～-1.6, -1.6～-0.8, -0.8～0, 0～0.8, 0.8～1.6, 1.6～2.4]
    ・カート速度：[-3～-2, -2～-1, -1～0, 0～1, 1～2, 2～3]
    ・ポール角度：[-0.3-0.2, -0.2～-0.1, -0.1～0, 0～0.1, 0.1～0.2, 0.2～0.3]（-54°～54°）
    ・ポール速度：[-3～-2, -2～-1, -1～0, 0～1, 1～2, 2～3]
したがってQテーブルは、状態数(6^4=1296)×行動選択肢(2)で2592マスのテーブルとなる
'''
import gym
import numpy as np

env = gym.make('CartPole-v0') # 倒立振子問題の環境を作成
# 状態数と行動選択肢を確認
obs_n, act_n = env.observation_space.shape[0], env.action_space.n
print('observation space: %d, action space: %d' % (obs_n, act_n))

max_steps = 200 # 1試行の最大ステップ数
episodes = 1000 # 試行回数
digitized = 6 # 状態変数の分割数
# 今回のQテーブルの初期状態は-1～1の一様乱数とする
q_table = np.random.uniform(low=-1, high=1, size=(digitized**obs_n, act_n))

# 4つの状態の組み合わせがQテーブルのどの行に該当するか計算
def digitize_state(obs):
    # カート位置、カート速度、ポール角度、ポール速度
    p, v, a, w = obs
    # numpy.linspace(min, max, n)
    ## min～maxをn分割した等差数列を生成
    ### numpy.digitizeでカテゴライズする場合、
    ### 必要分割数+1で分割した等差数列の頭と尾を抜いた数列を指定しないと、
    ### 想定する 0～n-1 のインデックスが得られない
    pl = np.linspace(-2.4, 2.4, digitized+1)[1:-1]
    vl = np.linspace(-3.0, 3.0, digitized+1)[1:-1]
    al = np.linspace(-0.3, 0.3, digitized+1)[1:-1]
    wl = np.linspace(-3.0, 3.0, digitized+1)[1:-1]
    # numpy.digitize(val, [array])
    ## val値が数列中のどこに該当するかのインデックスを返す
    '''
    例：digitize(val, [-1.0, 0.0, 1.0])
        val=-1.5 -> 0
        val=-1.0 -> 1
        val= 0.0 -> 2
        val= 1.0 -> 3
    '''
    pn = np.digitize(p, pl)
    vn = np.digitize(v, vl)
    an = np.digitize(a, al)
    wn = np.digitize(w, wl)
    # Qテーブルのどの行に該当するか変換
    return pn + vn*digitized + an*(digitized**2) + wn*(digitized**3)

# ε-グリーディー法により次の行動を決定
def get_action(next_state, episode):
    epsion = 0.5 * (0.99 ** episode) # 徐々にεを小さくする＝最適行動をとる確率を上げる
    if epsion <= np.random.uniform(0, 1):  # εが0～1の一様乱数以下なら最適行動をとる
        next_action = np.argmax(q_table[next_state]) # Q[t+1]が最大のインデックス(=行動)
    else: # それ以外は乱数行動
        next_action = np.random.randint(2)
    return next_action

# Qテーブルの更新処理
def update_Qtable(state, action, reward, next_state):
    alpha, gamma = 0.5, 0.99
    # Q(s[t], a[t]) <- (1 - α)・Q(s[t], a[t]) + α・(r[t+1] + γ・max(Q[t+1]))
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * max(q_table[next_state]))

# メイン処理
for episode in range(episodes):
    # 環境の初期化
    obs = env.reset()
    state = digitize_state(obs)
    action = np.argmax(q_table[state]) # とりあえず最初の行動は最大Q値の行動とする
    episode_reward = 0
    
    for t in range(max_steps): # 1試行の行動ループ
        env.render() # 画面描画
        obs, reward, done, info = env.step(action) # 1ステップ実行
        if done and t < max_steps-1: # 200ステップ終わる前にゲームが終了した場合
            reward -= max_steps # 罰則として-200
        episode_reward += reward # 1試行中に与えられた報酬を加算
        next_state = digitize_state(obs) # 次の状態を離散値に変換
        update_Qtable(state, action, reward, next_state) # 行動の結果からQテーブル更新
        action = get_action(next_state, episode) # 次の行動を選択
        state = next_state # 状態を次の状態へ更新
        if done:
            break # ゲーム終了と同時に1試行完了
    
    print('episode %d: total reward %d' % (episode+1, episode_reward))

env.close()

# Qテーブルを保存しておく
np.savetxt('06.Qtable.txt', q_table)