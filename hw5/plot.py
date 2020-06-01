import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('squareSum.csv', header=None)

y = df[0].tolist()
x = range(1, len(y)+1)

plt.plot(x, y)

plt.xlabel('K')
plt.ylabel('Sum of Squares')
plt.title('Total Within Group Sum of Squares')
plt.xticks(x)
plt.yticks(y)

plt.savefig('plots/q2_5_3.png')
plt.show()

p1 = pd.read_csv('p1.csv', header=None)
p2 = pd.read_csv('p2.csv', header=None)
p3 = pd.read_csv('p3.csv', header=None)

p1 = p1[0].tolist()
p2 = p2[0].tolist()
p3 = p3[0].tolist()
x = range(1, len(p1)+1)

plt.plot(x, p1)
plt.plot(x, p2)
plt.plot(x, p3)

plt.legend(['p1', 'p2', 'p3'])
plt.xlabel('K')
plt.ylabel('p values')
plt.title('Pair-counting Measures')
plt.xticks(x)

plt.savefig('plots/q2_5_4.png')
plt.show()
