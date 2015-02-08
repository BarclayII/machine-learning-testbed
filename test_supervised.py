#! /usr/bin/python -u
from DynNet import DynNet
from CatchBallEnvironment import CatchBallEnvironment as CBE
import numpy as NP
import pdb

NP.set_printoptions(linewidth=150)

e = CBE()
n = DynNet(learningmodel='supervised', save_filename='DynNetSup.pickle')

n.train(e)
