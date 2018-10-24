import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import collections
import random
mem = collections.namedtuple(
    'Mem', ['state', 'action', 'reward', 'future_state'])


class experience_replay:
    def __init__(self, N):
        self.i = 0
        self.replay = []
        self.mem_len = N

    def add_mem(self, state, action, reward, future_state):
        if len(self.replay) < self.mem_len:
            self.replay.append(mem(state, action, reward, future_state))
            self.i = (self.i+1) % self.mem_len
            return
        else:
            self.replay[self.i] = mem(state, action, reward, future_state)
            self.i = (self.i+1) % self.mem_len

    def sample_batch(self, batch_size):
        return random.sample(self.replay, batch_size)
