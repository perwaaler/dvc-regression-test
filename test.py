import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("data/data.csv")

print(df)

# x = np.array([0, 1, 2, 3])
# y = np.array([-1, 0.2, 0.9, 2.1])

p = 1.5

x = df.x1
A = np.vstack([x, x**p, np.ones(len(x))]).T
y = df.y
m = np.linalg.lstsq(A, y)

plt.plot(x,y - A@m[0].T,'*')