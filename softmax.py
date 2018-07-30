#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt

a = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]

a_exp = [math.exp(i) for i in a]

sigmoid = [(1 / (1 + math.exp(-i))) for i in a]

print sum(sigmoid)

s_a_exp = sum(a_exp)


softmax_a = [i / s_a_exp for i in a_exp]
print sum(softmax_a)

plt.figure(1)
plt.plot(softmax_a)
plt.plot(sigmoid)
plt.show()
