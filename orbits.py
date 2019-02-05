import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats

G = 0.0005

x = np.array([[0.,   0],
              [100, 0],
              [110, 0]])
v = np.array([[0,    0],
               [0, 0.02],
               [0, 0.030]])
m = np.array([100, 3, 1])

def update(x, v, m):
    x += v
    dx = x[:, None, :] - x
    dist = (dx ** 2).sum(axis=2) ** 1.5
    np.fill_diagonal(dist, 1)
    v += (G * m[:, None, None] * dx / dist[:, :, None]).sum(axis=0)


fig, ax = plt.subplots()

for t in range(30000):
    update(x, v, m)
    if t % 30 == 0:
        ax.plot(x[:, 0], x[:, 1], 'k.', ms=3)

ax.set_aspect('equal')
plt.show()
