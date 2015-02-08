#! /usr/bin/python -u
from DynNet import DynNet, identity
from CatchBallEnvironment import CatchBallEnvironment as CBE
import numpy as NP
import pdb

NP.set_printoptions(linewidth=150)

e = CBE()
n = DynNet(learningmodel='reinforce_sum', learning_rate=0.005, load_filename='DynNet.pickle', rate_decay_fn=identity,
           location_xvar=0.01, location_yvar=0.01, learning_mode='replay')

n.train(e)

