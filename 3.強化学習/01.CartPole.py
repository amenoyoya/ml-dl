import gym
import numpy as np

env = gym.make('CartPole-v0')
'''
# CartPole
カートの上に連結されているポールを倒さないように、カートを左右に動かすゲーム
ポールが一定の角度以上倒れてしまうとゲームオーバー

## 状態
環境から出力される状態sは以下の表の4つの変数からなる
    番号	名前        最小値  最大値
     0	 カートの位置	 -2.4	 2.4
     1	 カートの速度	 -inf	 inf
     2	 ポールの角度	 -41.8°	 41.8°
     3	 ポールの速度	 -inf	 inf

## 行動
ある状態sからとりうる行動A(s)
    番号	名前
     0	左にカートを押す
     1	右にカートを押す
'''

obs = env.reset() # 状態の初期化

for k in range(100): # ゲームを100ステップ実行
    env.render() # ゲーム画面描画
    act = np.random.randint(1) # 行動は乱数(0 or 1)で決定
    # 行動を起こし、その直後の[状態, 報酬, ゲームが終了したかどうか, 情報]を取得
    obs, reward, done, info = env.step(act)

env.close()