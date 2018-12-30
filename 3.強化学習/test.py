import numpy as np

for i in [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]:
    print('%f is in [%d] of [-1, 0, 1]' % (i, np.digitize(i, [-1, 0, 1])))