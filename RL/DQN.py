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
            img_ls.append((1/255)*cnn.composed(img).squeeze())
            env.step(i % 2)
    else:
        for i in range(frame_skip):
            img = env.render(mode='rgb_array')
            img_ls.append((1/255)*cnn.composed(img).squeeze())
            env.step(act)
    return torch.stack(img_ls).unsqueeze(0)


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


def fill_uniform_state_buf(env, frame_skip):
    uniform_state = []
    for eps in range(100):
        env.reset()
        while True:
            uniform_state.append(create_new_batch(
                env, frame_skip, np.random.randint(0, 2)).squeeze(0))
            _, _, done, _ = env.step(1)
            if done:
                break
    return uniform_state


def main():
    D = 1e6  # Amnt in replay memory
    M = 300000  # Number of epsiodes to run
    T = 50  # Number of steps to take maximum
    epsilon = 1  # Choose random actions prob
    save_iter = 100
    gamma = .99  # Forgetting factor of the past
    batch_size = 32  # Number of elements to sample from replay memory
    num_frames = 0  # Counter for the number of frames
    frame_skip = 4  # Number of frames to wait before selecting a new action
    env = gym.make('CartPole-v0')
    er = erp.experience_replay(D)
    reward_ls = []
    loss_fn = torch.nn.MSELoss()
    env.reset()
    writer = SummaryWriter()
    loss = 1
    uniform_state = torch.load('uniform_init.pt')
    # uniform_state = fill_uniform_state_buf(env, frame_skip)
    # torch.save(torch.stack(uniform_state), 'uniform_init.pt')
    # writer.add_graph(cnn.model, torch.autograd.Variable(
    #    torch.Tensor(4, 84, 84)))
    for eps in range(M):
        env.reset()
        batch = create_new_batch(env, frame_skip, act=-1)
        reward_gl = 0
        print(eps, epsilon, num_frames)
        for t in range(T):
            if random.random() < epsilon:
                act = env.action_space.sample()
            else:
                with torch.no_grad():
                    Q = cnn.model(batch)
                    act = torch.argmax(Q).item()
            _, reward, done, _ = env.step(act)
            reward = bound_reward(reward)
            # reward_ls.append(reward)
            num_frames += frame_skip
            reward_gl += reward
            if done:
                er.add_mem(batch, act, reward, False)
            else:
                new_batch = create_new_batch(env, frame_skip, act=act)
                er.add_mem(batch, act, reward, new_batch)
                batch = new_batch

            if len(er.replay) > batch_size:
                state_batch, action_batch, reward_batch, next_state_batch = er.sample_batch(
                    batch_size)
                state_back = torch.stack(state_batch)
                act = torch.tensor(action_batch)
                rew = torch.tensor(reward_batch)
                mask_nd = torch.tensor([
                    type(mem) == torch.Tensor for mem in next_state_batch])
                non_final_next_states = torch.stack([s.squeeze(0) for i, s in enumerate(next_state_batch)
                                                     if mask_nd[i]])
                Q = cnn.model(state_back).gather(
                    1, act.unsqueeze(1))
                Q_opt = torch.zeros(batch_size)
                Q_opt[mask_nd] = cnn.model(
                    non_final_next_states).detach().max(1)[0]
                expected_reward = rew.float()+gamma*Q_opt
                loss = loss_fn(Q.squeeze(1), expected_reward)
                print(loss)
                cnn.optimizer.zero_grad()
                loss.backward()
                cnn.optimizer.step()

            if done:
                break
        writer.add_scalar('data/eps_len', t, num_frames)
        writer.add_scalar('data/reward', reward_gl, num_frames)
        writer.add_scalar('data/eps', epsilon, num_frames)
        with torch.no_grad():
            writer.add_scalar(
                'data/avg_Q', torch.mean(torch.max(cnn.model(uniform_state), 1)[0]), num_frames)
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
