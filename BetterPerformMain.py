'''
Main Program
'''
import gym
from BetterPerformBrain import PolicyGradient
import matplotlib.pyplot as plt
import pdb
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
        


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    """ placement of this function is to be deteremined """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    # ::2 even indexing downsampling
    # output index 0 2 4 6 8... in the array as a new array
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.long).ravel() # return the image that is flattenedd into 1D array


env = gym.make('Pong-v0')
# env.seed(1)
env = env.unwrapped
D = 80*80
RL = PolicyGradient(
	n_a = env.action_space.n, 
	n_f = D
)
if USE_CUDA:
    RL.NN.cuda()
# 1000 episodes
# to be modified such that each point is an episode


RENDER = True
running_reward = None
RL.NN.load_state_dict(torch.load('PongNetParameter.pkl', map_location=lambda storage, loc: storage))


for i in range(50000):
    
    observation = env.reset()
    totalR = 0
    s_previous = None
    while True:
        if RENDER: env.render()
        s_current = prepro(observation)
        s = s_current - s_previous if s_previous is not None else np.zeros(D)
        s_previous = s_current
        # sprevious = scurrent
        a = RL.choose_action(s)
        observation, r, done, info = env.step(a)
        RL.store_transition(s,a,r)

        '''if r != 0:
            totalR += r
            # pdb.set_trace()
            pdb.set_trace()
            vt = RL.learn()
            #   print('Gain = ',RL.loss_return)
        '''
        totalR += r
        
        #if running_reward is not None and running_reward > 0.9:
        #    RENDER = True

        if done:
            # pdb.set_trace()
            RL.learn()
            running_reward = totalR if running_reward is None else running_reward*0.99+ totalR * 0.01
            # print("episode:", i, "  reward:", totalR, ' running mean: ',running_reward)
            print('resetting env. episode %.0f reward total was %.5f. running mean: %.5f' % (i+1,totalR, running_reward))
            if running_reward > 0.9 and i > 500: RENDER = True
            if i>0 and i%5 ==0: torch.save(RL.NN.state_dict(),'PongNetParameter.pkl')

            break

        # type: np/numeric, non-tensor/variable like
        # s = s_

