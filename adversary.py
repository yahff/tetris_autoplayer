from random import Random

from board import Shape
from exceptions import BlockLimitException


class Adversary:
    def choose_block(self, board):
        raise NotImplementedError


class RandomAdversary(Adversary):
    def __init__(self, seed, blocks=None):
        self.random = Random(seed)
        self.blocks = blocks

    def choose_block(self, board):
        if self.blocks is not None:
            if self.blocks == 0:
                raise BlockLimitException()
            else:
                self.blocks -= 1
        return self.random.choice(list(Shape)[:-1])
