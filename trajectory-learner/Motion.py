
import numpy as NP
import numpy.random as RNG

def expand(val, maxval, minval = 0):
    dim = maxval - minval
    sig = NP.zeros((dim,))
    sig[int(NP.floor(val - minval))] = 1
    return sig

def restore(sig, maxval, minval = 0):
    return NP.argmax(sig) + minval

def probabilistic_restore(sig, maxval, minval = 0):
    return RNG.choice(maxval - minval, p = sig) + minval

def uniform_motion(map_width, map_height, posX=None, posY=None, velX=None, velY=None):
    if posX == None or posY == None:
        posX = map_width / 2 + 0.5
        posY = map_height / 2 + 0.5
    if velX == None or velY == None:
        start_velocity = 1.0
        angle = RNG.ranf() * 2 * NP.pi
        velX = start_velocity * NP.cos(angle)
        velY = start_velocity * NP.sin(angle)
    X = posX
    Y = posY
    while True:
        a = yield (X, Y)
        if a != None:
            raise StopIteration
        X += velX
        Y += velY
        while (X < 0) or (X > map_width):
            if X < 0:
                X = -X
                velX = -velX
            if X > map_width:
                X = 2 * map_width - X
                velX = -velX
        while (Y < 0) or (Y > map_width):
            if Y < 0:
                Y = -Y
                velY = -velY
            if Y > map_width:
                Y = 2 * map_width - Y
                velY = -velY



def hyperbolic_motion(map_width, map_height, posX=None, posY=None, velX=None, velY=None, accX=None, accY=None):
    if posX == None or posY == None:
        posX = map_width / 2 + 0.5
        posY = map_height / 2 + 0.5
    if velX == None or velY == None:
        start_velocity = 1.0
        angle = RNG.ranf() * 2 * NP.pi
        velX = start_velocity * NP.cos(angle)
        velY = start_velocity * NP.sin(angle)
    if accX == None or accY == None:
        accX = (NP.floor(RNG.ranf() * 5) - 2.0) / 10.0
        accY = 0
    X = posX
    Y = posY
    while True:
        a = yield (X, Y)
        if a != None:
            raise StopIteration
        X += velX
        Y += velY
        velX += accX
        velY += accY
        while (X < 0) or (X > map_width):
            if X < 0:
                X = -X
                velX = -velX
            if X > map_width:
                X = 2 * map_width - X
                velX = -velX
        while (Y < 0) or (Y > map_width):
            if Y < 0:
                Y = -Y
                velY = -velY
            if Y > map_width:
                Y = 2 * map_width - Y
                velY = -velY



def circular_motion(map_width, map_height, centerX=None, centerY=None, radius=10, start_angle=0, angular_velocity=0.1):
    if centerX == None or centerY == None:
        centerX = map_width / 2 + 0.5
        centerY = map_height / 2 + 0.5
    if (centerX - radius < 0) or (centerX + radius > map_width) or (centerY - radius < 0) or (centerY + radius > map_height):
        raise StopIteration
    while True:
        X = centerX + radius * NP.cos(start_angle)
        Y = centerX + radius * NP.sin(start_angle)
        a = yield (X, Y)
        if a != None:
            raise StopIteration
        start_angle += angular_velocity
