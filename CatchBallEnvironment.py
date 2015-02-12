"""
Catch-ball game environment class.

This environment is based on the Jun. 2014 paper from DeepMind.
"""

import numpy as NP

# custom classes
class Ball:
    """
    The ball in the catch-ball game.
    """
    def __init__(self, env, posX, posY, velX, velY):
        """
        Initializes a ball with given position and velocity in pixels
        or pixels per frame, in the given environment @env.
        The values need not be integers.
        """
        self.posX = posX
        self.posY = posY
        self.velX = velX
        self.velY = velY
        self._envsize = env.size()

    def tick(self):
        """
        Moves the ball a frame further.
        Returns true if the ball is still inside the game frame, or false
        if the ball is beyond the bottom.
        """
        # If the ball is already at (or beyond) the bottom, return
        # directly
        if self.posY < 0:
            return False
        self.posX += self.velX
        self.posY += self.velY
        #self.velX += NP.random.uniform(-0.2, 0.2)
        #self.velY += NP.random.uniform(-0.2, 0.0)
        # The ball will bounce if colliding with the vertical walls
        while self.posX < 0 or self.posX > self._envsize:
            if self.posX < 0:
                self.posX = -self.posX
                self.velX = -self.velX
            elif self.posX > self._envsize:
                self.posX = 2 * self._envsize - self.posX
                self.velX = -self.velX
        return True

class Board:
    """
    The board in the catch-ball game.
    """
    def __init__(self, env, posX):
        """
        Initializes a board with given position inside environment
        @env.  The board is always 1-pixel high, and resides on
        y = 0.
        """
        self._envsize = env.size()
        self.posX = posX
        self.posY = 0

    def moveLeft(self):
        self.posX = (self.posX - 1) if (self.posX > 0) else 0

    def moveRight(self):
        self.posX = (self.posX + 1) if (self.posX + 2 < self._envsize) \
                else self._envsize - 2

    def stay(self):
        pass


class CatchBallEnvironment:
    """
    Catch-ball game environment.
    """

    def __init__(self, size = 24):
        """
        This constructor does *not* start a game.  It merely sets up some basic
        parameters.
        """
        self._size = size
        self.M = NP.zeros((size, size))

        # The ball and board information is invisible to the agent.  The agent
        # can only observe the environment matrix, i.e. image.
        self._ball = None
        self._board = None

    def size(self):
        return self._size

    def start(self):
        angle = (NP.random.ranf() * 0.8 - 0.4) * NP.pi
        ball_velocity = 1.0
        ball_startPos = NP.random.ranf() * self._size
        board_startPos = NP.random.randint(self._size - 1)
        self._ball = Ball(self, ball_startPos, self._size - 1,
                ball_velocity * NP.sin(angle), -ball_velocity * NP.cos(angle))
        #self._board = Board(self, board_startPos)
        self._refresh()

    def done(self):
        """
        Returns true if the board had successfully catched the ball.
        Returns false if the ball had passed the bottom.
        Returns none if the game has not finished yet.
        """
        #if 0 < self._ball.posY < 1 and \
        #        self._board.posX < self._ball.posX < self._board.posX + 2:
        #    return True
        if self._ball.posY < 0:
            return True
        else:
            return False

    def tick(self, action = 0):
        """
        Proceed the game by one frame, and moves the board according to
        @action.
        If @action is negative, the board is moved left.
        If @action is positive, it is moved right.
        If @action is zero, the board stays at the original position.
        """
        self._ball.tick()
        #if action < 0:
        #    self._board.moveLeft()
        #elif action > 0:
        #    self._board.moveRight()
        #else:
        #    self._board.stay()
        self._refresh()

    def is_tracking(self, loc, gw):
        """
        Determine whether the glimpse with width @gw is currently
        tracking the ball.
        """
        x,y = loc[0], loc[1]
        x = round(x - 0.5) + 0.5
        y = round(y - 0.5) + 0.5

        left = int((2 * x - gw + 1) / 2)
        right = int((2 * x + gw - 1) / 2)
        top = int((2 * y - gw + 1) / 2)
        bottom = int((2 * y + gw - 1) / 2)

        return (left <= self._ball.posX <= right) and \
                (top <= self._ball.posY <= bottom)

    def _refresh(self):
        self.M.fill(0)
        self.M[NP.int(NP.floor(self._ball.posX)),
                NP.int(NP.floor(self._ball.posY))] = 255
        #self.M[NP.int(NP.floor(self._board.posX)), 0] = 255
        #self.M[NP.int(NP.floor(self._board.posX)) + 1, 0] = 255
