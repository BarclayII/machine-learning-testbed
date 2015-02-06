#! /usr/bin/python -u
from DynNet import DynNet
from CatchBallEnvironment import CatchBallEnvironment as CBE
import numpy as NP
import pdb

NP.set_printoptions(linewidth=150)

e = CBE()
n = DynNet(learningmodel='reinforce_sum', learning_rate=0.0005)

n.train(e)

