#!/usr/bin/env python

import numpy as np
import math
import random


class Bandit:
    def __init__(self):
        #self.arm_values = np.random.normal(0, 1, 10)
        self.arm_values = np.array([2, 4, 1, 7, 5, 6, 10, 3, 8, 9])
        self.K = np.zeros(10)
        #self.est_values = np.zeros(10)
        self.est_values = np.array([5] * 10)
        self.temperature = 15

    def get_reward(self, action):
        noise = np.random.normal(0, 1)
        reward = self.arm_values[action] + noise

        return reward

    def choose_eps_greedy(self, epsilon):
        rand_num = np.random.random()

        if epsilon > rand_num:
            return np.random.randint(10)
        else:
            return np.argmax(self.est_values)

    def softmax(self):

        z = sum(math.exp(v / self.temperature) for v in self.est_values)
        probs = [math.exp(v / self.temperature) / z for v in self.est_values]

        return self.categorical_draw(probs)

    def categorical_draw(self, probs):

        threshold = random.random()

        cum_sum = 0.0

        for i in range(len(probs)):

            prob = probs[i]
            cum_sum += prob

            if cum_sum > threshold:
                return i

        return len(probs) - 1

    def update_est(self, action, reward):
        self.K[action] += 1
        #alpha = 1.0 / self.K[action]
        alpha = 0.03
        self.est_values[action] += alpha * (reward - self.est_values[action])


def experiment(bandit, Npulls, epsilon):
    history = []

    for i in range(Npulls):
        print 'running....'
        action = bandit.choose_eps_greedy(epsilon)
        #action = bandit.softmax()
        R = bandit.get_reward(action)
        bandit.update_est(action, R)
        history.append(R)

    return np.array(history), bandit.K, bandit.est_values


Nexp = 2000
Npulls = 3000
avg_outcome_eps0p0 = np.zeros(Npulls)
avg_outcome_eps0p01 = np.zeros(Npulls)
avg_outcome_eps0p1 = np.zeros(Npulls)

k_value0 = np.zeros(10)
k_value01 = np.zeros(10)
k_value1 = np.zeros(10)

est_values0 = np.zeros(10)
est_values01 = np.zeros(10)
est_values1 = np.zeros(10)


for i in range(Nexp):
    bandit = Bandit()
    #avg_outcome_eps0p0 += experiment(bandit, Npulls, 0.0)
    value0 = experiment(bandit, Npulls, 0.0)
    avg_outcome_eps0p0 += value0[0]
    k_value0 += value0[1]
    est_values0 += value0[2]

    bandit = Bandit()
    #avg_outcome_eps0p01 += experiment(bandit, Npulls, 0.01)
    value01 = experiment(bandit, Npulls, 0.01)
    avg_outcome_eps0p01 += value01[0]
    k_value01 += value01[1]
    est_values01 += value01[2]

    bandit = Bandit()
    #avg_outcome_eps0p1 += experiment(bandit, Npulls, 0.1)
    value1 = experiment(bandit, Npulls, 0.1)
    avg_outcome_eps0p1 += value1[0]
    k_value1 += value1[1]
    est_values1 += value1[2]


avg_outcome_eps0p0 /= np.float(Nexp)
avg_outcome_eps0p01 /= np.float(Nexp)
avg_outcome_eps0p1 /= np.float(Nexp)

k_value0 /= np.float(Nexp)
k_value01 /= np.float(Nexp)
k_value1 /= np.float(Nexp)

est_values0 /= np.float(Nexp)
est_values01 /= np.float(Nexp)
est_values1 /= np.float(Nexp)


print est_values0
print est_values01
print est_values1


# plot results
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(avg_outcome_eps0p0, label="eps = 0.0")
plt.plot(avg_outcome_eps0p01, label="eps = 0.01")
plt.plot(avg_outcome_eps0p1, label="eps = 0.1")
plt.ylim(-1.0, 20.0)
plt.legend()

plt.figure(2)
plt.bar(np.arange(10), k_value0, color='b', label='0')
plt.xticks(np.arange(10) + 0.35 / 2, ('arm1', 'arm2', 'arm3', 'arm4', 'arm5', 'arm6', 'arm7', 'arm8', 'arm9', 'arm10'))

plt.figure(3)
plt.bar(np.arange(10), k_value01, color='g', label='01')
plt.xticks(np.arange(10) + 0.35 / 2, ('arm1', 'arm2', 'arm3', 'arm4', 'arm5', 'arm6', 'arm7', 'arm8', 'arm9', 'arm10'))
plt.figure(4)
plt.bar(np.arange(10), k_value1, color='r', label='1')
plt.xticks(np.arange(10) + 0.35 / 2, ('arm1', 'arm2', 'arm3', 'arm4', 'arm5', 'arm6', 'arm7', 'arm8', 'arm9', 'arm10'))


plt.show()
