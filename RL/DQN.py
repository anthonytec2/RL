import cnn
import erp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import gym
import collections
import random
from tensorboardX import SummaryWriter


def create_new_batch(env, frame_skip, act=-1):
    img_ls = []
    done = False
    if act == -1:
        for i in range(frame_skip):
            img = env.render(mode='rgb_array')
            img_ls.append(cnn.composed(img))
            env.step(i % 2)
    else:
        for i in range(frame_skip):
            img = env.render(mode='rgb_array')
            img_ls.append(cnn.composed(img))
            env.step(act)
    return torch.stack(img_ls)


def bound_reward(reward):
    if reward == 0:
        return 0
    elif reward > 0:
        return 1
    elif reward < 0:
        return -1


def eps_anneal(epsilon):
    if epsilon < .1:
        epsilon = .1
    else:
        epsilon -= (1-.1)/(1e3)
    return epsilon


def main():
    D = 1e6  # Amnt in replay memory
    M = 300000  # Number of epsiodes to run
    T = 50  # Number of steps to take maximum
    epsilon = 1  # Choose random actions prob
    save_iter = 100
    gamma = .9  # Forgetting factor of the past
    batch_size = 32  # Number of elements to sample from replay memory
    num_frames = 0  # Counter for the number of frames
    frame_skip = 4  # Number of frames to wait before selecting a new action
    env = gym.make('CartPole-v0')
    er = erp.experience_replay(D)
    reward_ls = []
    env.reset()
    writer = SummaryWriter()
    writer.add_graph(cnn.model, torch.autograd.Variable(
        torch.Tensor(4, 1, 84, 84)))
    for eps in range(M):
        env.reset()
        batch = create_new_batch(env, frame_skip, act=-1)
        reward_gl = 0
        max_q = 0
        print(eps, epsilon, num_frames)
        for t in range(T):
            if random.random() < epsilon:
                act = env.action_space.sample()
            else:
                with torch.no_grad():
                    Q = cnn.model(batch)
                    if torch.max(Q) > max_q:
                        max_q = torch.max(Q).item()
                    act = torch.argmax(Q).item()
            _, reward, done, _ = env.step(act)
            num_frames += frame_skip
            reward = bound_reward(reward)
            reward_gl += reward
            reward_ls.append(reward)
            if done:
                er.add_mem(batch, act, reward, False)
            else:
                new_batch = create_new_batch(env, frame_skip, act=act)
                er.add_mem(batch, act, reward, new_batch)
                batch = new_batch

            if len(er.replay) > batch_size:
                cnn.optimizer.zero_grad()
                rnd_mini_batch = er.sample_batch(32)
                for memory in rnd_mini_batch:
                    if hasattr(memory.future_state, "shape"):
                        with torch.no_grad():
                            y = memory.reward+gamma * \
                                torch.max(cnn.model(memory.future_state))
                    else:
                        y = memory.reward
                    loss = (y-cnn.model(memory.state)[act])**2
                    loss.backward()
                cnn.optimizer.step()
            if done:
                break
        writer.add_scalar('data/eps_len', t, num_frames)
        writer.add_scalar('data/q', t, max_q)
        writer.add_scalar('data/reward', reward_gl, num_frames)
        writer.add_scalar('data/eps', epsilon, num_frames)
        epsilon = eps_anneal(epsilon)
        if eps % save_iter == 0:
            torch.save({
                'epoch': num_frames,
                'model_state_dict': cnn.model.state_dict(),
                'optimizer_state_dict': cnn.optimizer.state_dict()},
                'model/model.pt')
    env.close()
    writer.close()


if __name__ == '__main__':
    main()