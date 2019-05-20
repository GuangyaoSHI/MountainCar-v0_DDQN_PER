import time
import random
import pickle
import os
import sys
import math

import torch
import torch.optim as optim
import torch.nn as nn

from QNet import QNet
from replay_memory import ReplayMemory
import utils

import numpy as np
import cv2
import gym

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

def prepare_state(state):
    ret = np.array([(state[0]+.3)/.9, state[1]/.07])
    ret = np.expand_dims(ret, axis=0)
    return ret

if __name__ == '__main__':
    #Save/load dirs
    models_dir = './models'
    state_file = 'state'
    out_dir = os.path.join(models_dir, str(int(time.time())))

    #Engine parameters
    env = gym.make('MountainCar-v0')

    num_actions = 3

    #Initing neural networks and loading previos state, if exists
    Q = QNet(num_actions=num_actions)
    Q_targ = QNet(num_actions=num_actions)

    prev_state = None
    if os.listdir(models_dir) == []:
        Q.apply(utils.init_weights)

        Q_targ.load_state_dict(Q.state_dict())
    else:
        weights_file_path, state_file_path = utils.find_prev_state_files(models_dir, state_file)

        weights = torch.load(weights_file_path)
        prev_state = torch.load(state_file_path)

        Q.load_state_dict(weights['Q'])
        Q_targ.load_state_dict(weights['Q_targ'])

    #Learn params
    gamma = 0.99

    #Hyperparams
    frame_count = 40000
    eps_decay_time = 0.5

    eps_start = 1
    eps_end = 0.05
    l = -math.log(eps_end) / (frame_count * eps_decay_time)

    replay_mem = ReplayMemory(max_size=500000, alpha=0.5, eps=0.0) if prev_state is None else prev_state['replay_mem']

    episode_depth = 10000

    batch_size = 32

    Q_targ_update_freq = 300

    save_freq = frame_count / 100

    #Episode loop
    curr_eps = eps_start if prev_state is None else prev_state['end_eps']

    episode_num = 0 if prev_state is None else (prev_state['end_episode'] + 1)
    curr_frame_count = 0 if prev_state is None else prev_state['curr_frame_count']

    #Optimizer init
    if prev_state is None:
        optimizer = optim.Adam(Q.parameters(), lr=0.0000625, eps=1.5e-4)
    else:
        optimizer = optim.Adam(Q.parameters())
        optimizer.load_state_dict(prev_state['optimizer'])

    while True:
        if curr_frame_count > frame_count:
            break

        print()
        print('===================================================')
        print('Episode number:', episode_num)
        print('Frames processed:', round(curr_frame_count / frame_count * 100, 2), '%')

        losses = []

        env.reset()
        env.render()

        raw_state, reward, done, info = env.step(1)
        start_pos = raw_state[0]

        curr_state = prepare_state(raw_state)
        reward = 0
        
        Q_out = Q(FloatTensor(curr_state)).to('cpu').detach().numpy()[0]

        for t in range(episode_depth):
            #
            # Acting
            #

            if done:
                break

            print(t, ' ', end='')
            sys.stdout.flush()

            action = None
            if random.random() < curr_eps:
                action = random.randint(0, num_actions - 1)
            else:
                action = np.argmax(Q_out)

            new_state, reward, done, info = env.step(action)
            env.render()

            reward = abs(new_state[0] - start_pos)

            loss = reward - np.amax(Q_out)

            if not done:
                new_state = prepare_state(new_state)

                Q_out = Q(FloatTensor(new_state)).to('cpu').detach().numpy()[0]
                Q_targ_out = Q_targ(FloatTensor(new_state)).to('cpu').detach().numpy()[0]

                loss += gamma * Q_targ_out[np.argmax(Q_out)]

            loss = abs(loss)

            #print(curr_state.shape)
            replay_mem.add_element((curr_state, action, reward, new_state), loss)

            curr_state = new_state

            #
            #Learning
            #

            sarses = replay_mem.get_batch(batch_size)

            #Targets

            Q_true = []

            #print()
            #print('+++++++++++++++++++++++++++')
            #for x in [sars[3] for sars in sarses]:
            #    print(x)

            batch = np.concatenate([(sars[3] if len(np.array(sars[3]).shape) == 2 else [sars[3]]) for sars in sarses], axis=0)
            batch = FloatTensor(batch)

            actions = np.argmax(Q(batch).to('cpu').detach().numpy(), axis=1)
            Q_targ_out = Q_targ(batch).to('cpu').detach().numpy()
            Q_true = np.array([Q_targ_out[i][actions[i]] for i in range(Q_targ_out.shape[0])])

            rs = np.array([sars[2] for sars in sarses])
            Q_true = rs + gamma * Q_true

            #Predictions

            batch = np.concatenate([sars[0] for sars in sarses], axis=0)
            batch = FloatTensor(batch)
            Q_pred = Q(batch)
            mask = np.array([utils.dirac_delta(i=sars[1], n=num_actions) for sars in sarses])
            mask = ByteTensor(mask)
            Q_pred = torch.masked_select(Q_pred, mask=mask)

            #Updating replay memory

            replay_mem.update(np.abs(Q_true - Q_pred.to('cpu').detach().numpy()))

            #Learning

            loss = nn.SmoothL1Loss()(Q_pred, FloatTensor(Q_true))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.to('cpu').detach().numpy()))

            #
            #Updating frame counter
            #

            curr_frame_count += 1

            #
            #Updating eps
            #

            curr_eps = max(math.exp(-l * curr_frame_count), eps_end)

            #
            #Updating Q_targ if needed
            #

            if curr_frame_count % Q_targ_update_freq == 0:
                print('Updating Q_targ')

                Q_targ.load_state_dict(Q.state_dict())

            #
            #Saving if needed
            #

            if curr_frame_count % save_freq == 0:
                print('Saving model')

                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                torch.save({'Q': Q.state_dict(), 'Q_targ' : Q_targ.state_dict()}, os.path.join(out_dir, str(curr_frame_count)))

                state_file_path = os.path.join(out_dir, state_file)

                if os.path.exists(state_file_path):
                    bak_state_file_path = os.path.join(out_dir, state_file + '_bak')

                    if os.path.exists(bak_state_file_path):
                        os.remove(bak_state_file_path)
                    os.rename(state_file_path, bak_state_file_path)

                torch.save({'replay_mem' : replay_mem,
                    'end_eps' : curr_eps,
                    'end_episode' : episode_num,
                    'curr_frame_count' : curr_frame_count,
                    'optimizer' : optimizer.state_dict()
                }, state_file_path)

        #
        #Other info
        #

        print()
        print('Average loss:', np.mean(losses))
        print('Replay memory size:', len(replay_mem))
        print('Eps:', curr_eps)

        #
        #Episode loop routine
        #

        episode_num += 1

    print()
    print('Done!')