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
        rnd_mini_batch = random.sample(self.replay, batch_size)
        state_batch = [mem.state.squeeze()
                       for mem in rnd_mini_batch]
        action_batch = [mem.action for mem in rnd_mini_batch]
        reward_batch = [mem.reward for mem in rnd_mini_batch]
        next_state_batch = [
            mem.future_state for mem in rnd_mini_batch]
        return state_batch, action_batch, reward_batch, next_state_batch
