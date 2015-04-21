#! /usr/bin/python -u
from TrajLearner import *
from Motion import *

import sys

training_size = 100000
steps = 30
map_width = 192
map_height = 192
vel_left = -map_width + 1
vel_top = -map_height + 1
vel_width = map_width - vel_left
vel_height = map_height - vel_top
lstm_units = (256,)
learn_rate = 0.05

NP.set_printoptions(linewidth=150)

def linear_log(x):
    return 0.001 * TT.log(1 + x) + 0.999 * x

print 'Initializing RNN...'
rnn = TrajectoryRNN((map_width, map_height, vel_width, vel_height), lstm_units, (map_width, map_height, vel_width, vel_height))
print 'Bootstrapping trainer...'
trainer = OnlineTrainer(rnn, learn_rate)
print 'Training start'

mean = 0
for i in range(0, training_size):
    gen = uniform_motion(map_width, map_height)
    sigX_seq = []
    sigY_seq = []
    velX_seq = []
    velY_seq = []
    for t in range(0, steps):
        X, Y = next(gen)
        sx = expand(X, map_width)
        sy = expand(Y, map_height)
        if t > 0:
            vx = expand(NP.floor(X) - NP.floor(_X), map_width, vel_left)
            vy = expand(NP.floor(Y) - NP.floor(_Y), map_height, vel_top)
        else:
            vx = expand(0, map_width, vel_left)
            vy = expand(0, map_height, vel_top)
        sigX_seq.append(sx)
        sigY_seq.append(sy)
        velX_seq.append(vx)
        velY_seq.append(vy)
        _X = X
        _Y = Y
    labelX_seq = list(sigX_seq)
    labelY_seq = list(sigY_seq)
    labelVX_seq = list(velX_seq)
    labelVY_seq = list(velY_seq)
    sigX_seq.pop()
    sigY_seq.pop()
    velX_seq.pop()
    velY_seq.pop()
    labelX_seq.pop(0)
    labelY_seq.pop(0)
    labelVX_seq.pop(0)
    labelVY_seq.pop(0)
    res = trainer.learn(sigX_seq, sigY_seq, velX_seq, velY_seq, labelX_seq, labelY_seq, labelVX_seq, labelVY_seq)
    cost = res[-1].mean()
    j = i % 200
    prev_mean = mean
    mean = (mean * j + cost) / (j + 1)
    print i, cost, mean, trainer.learn_rate.get_value()
    if i % 200 == 0:
        for i in range(0, steps - 1):
            print '\t', restore(labelX_seq[i], map_width), restore(labelY_seq[i], map_height), \
                    restore(labelVX_seq[i], map_width, vel_left), restore(labelVY_seq[i], map_height, vel_top), \
                    restore(res[0][i], map_width), restore(res[1][i], map_height), \
                    restore(res[2][i], map_width, vel_left), restore(res[3][i], map_height, vel_top)
