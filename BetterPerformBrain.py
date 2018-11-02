'''
Reinforcement Learning Brain
'''


import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
# np.random.seed(1)
# torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self, n_actions, n_features):
        # n_features indicate the number of elements in a flatten image
        super(Net, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.fc1 = nn.Linear(in_features= self.n_features, out_features= 200)
        self.fc2 = nn.Linear(in_features= 200, out_features=self.n_actions)
        # number of possible actions in pong game in relation to
        # network output position
        # I simply take all actions into considerations despite some 
        # are invalid
        
    def forward(self,x):
        # x is the flattened observation
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        p = F.softmax(x)
        return p


class PolicyGradient:
    def __init__(
            self,
            n_a, 
            n_f,
            LR=1e-4,
            GAMMA=0.99,
    ):
        
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.NN = Net(n_actions = n_a, n_features = n_f)
        # USE_CUDA = torch.cuda.is_available()
        # FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        # self.LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
        #if USE_CUDA:
        #    self.NN.cuda()
        self.optimizer = torch.optim.Adam(self.NN.parameters(),lr = LR)
        self.gamma = GAMMA
        self.n_a = n_a
        self.n_f = n_f


    
    def choose_action(self,env_in):
        env_in = Variable(torch.from_numpy(env_in).type(FloatTensor))
        actions_prob = self.NN.forward(env_in)
        action = np.random.choice(range(actions_prob.cpu().data.numpy().shape[0]), p = actions_prob.cpu().data.numpy().ravel())
        # greediness included
        return action
        # 直至呢度都冇错
    def store_transition(self,s,a,r):
        # store a difference frame betweeen 2 timestamps
        self.ep_obs.append(s)
        # store flattened array
        self.ep_as.append(a)
        self.ep_rs.append(r)
        '''
        these lists are 1D. Each element represents 1 single timestamp
        using np.vstack can sort out the corresponding actions, rewards, and obs
        at each timestamp
        ''' 
        
    def learn(self):
        '''be careful on the data type '''
        class_num = self.n_a
        # in one timestep, there are 6 classes (possible actions)
        batch_size = len(self.ep_as)
        # episodic batch, taking the whole episode 
        # as one single batch to train
        self.ep_obs = Variable(torch.from_numpy(np.vstack(self.ep_obs)).type(FloatTensor),requires_grad = True)
        self.ep_p = self.NN.forward(self.ep_obs)
        self.ep_as = np.vstack(self.ep_as)
        # up to here
        label = torch.from_numpy(self.ep_as).type(torch.LongTensor)
        discounted_ep_rs_norm = np.vstack(self._discount_and_norm_rewards())
        self.vt = Variable(torch.from_numpy(discounted_ep_rs_norm).type(FloatTensor),requires_grad = True)
        
        # neg_log_prop = torch.sum(-1*torch.log(self.ep_p) * Variable(torch.zeros(batch_size, class_num).scatter_(1, label, 1),requires_grad = False)) 
        # loss = torch.mean(neg_log_prop * self.vt)
        loss = torch.mean(-1*torch.log(self.ep_p)* self.vt * Variable(torch.zeros(batch_size, class_num).scatter_(1, label, 1).type(FloatTensor),requires_grad = True)) 
        
        


        self.optimizer.zero_grad()
        loss.backward()
        #debug_l = list(self.NN.parameters())
        #print(debug_l[0].grad.data.numpy()[debug_l[0].grad.data.numpy()>0.000001])
        self.optimizer.step()
        # clear the memory
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return None
        
        
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        # to be studied
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, np.vstack(self.ep_rs).size)):
            # change based on sample
            if self.ep_rs[t] != 0: running_add = 0
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    