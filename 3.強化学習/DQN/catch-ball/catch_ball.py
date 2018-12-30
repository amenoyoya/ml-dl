import numpy as np

class CatchBall:
  def __init__(self):
    self.screen_rows = self.screen_cols = 8
    self.player_length = 3
    self.enable_actions = (-1, 0, 1)
    # reset variables
    self.reset()
  
  ''' 変数リセット '''
  def reset(self):
    # reset player position
    self.player_row = self.screen_rows - 1
    self.player_col = np.random.randint(self.screen_cols - self.player_length)
    # reset ball position
    self.ball_row = 0
    self.ball_col = np.random.randint(self.screen_cols)
    # reset other variables
    self.reward = 0
    self.terminal = False
  
  ''' プレイヤーの行動を受けてゲームを更新 '''
  def update(self, action):
    """
    action:
       0: do nothing
      -1: move left
       1: move right
    """
    # update player position
    if action in self.enable_actions:
      self.player_col += action
      if self.player_col < 0:
        self.player_col = 0
      elif self.player_col > self.screen_cols - self.player_length:
        self.player_col = self.screen_cols - self.player_length
    
    # update ball position
    self.ball_row += 1
    
    self.reward = 0
    self.terminal = False
    # collision detection
    if self.ball_row == self.screen_rows - 1:
      self.terminal = True
    if self.player_col <= self.ball_col < self.player_col + self.player_length:
      # catch
      self.reward = 1
    else:
      # drop
      self.reward = -1
  
  ''' ゲームの状態を取得 '''
  def observe(self):
    # reset screen
    self.screen = np.zeros((self.screen_rows, self.screen_cols))
    # draw player
    self.screen[self.player_row, self.player_col : self.player_col + self.player_length] = 1
    # draw ball
    self.screen[self.ball_row, self.ball_col] = 1
    return self.screen, self.reward, self.terminal