import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from memory import Memory
import config as p


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.boxfc1=nn.Linear(p.BOX_NUM*6, 128)
        self.boxfc2=nn.Linear(128, 256)
        self.stepfc=nn.Linear(p.MAX_STEP, 256)
        
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 1024)
        self.action = nn.Linear(1024, p.ACTION_NUM)
        
        self.conv1[0].weight.data.normal_(0, 0.1)
        self.conv2[0].weight.data.normal_(0, 0.1)
        self.conv3[0].weight.data.normal_(0, 0.1)

        self.boxfc1.weight.data.normal_(0, 0.1)
        self.boxfc2.weight.data.normal_(0, 0.1)

        self.fc1.weight.data.normal_(0, 0.1)  
        self.fc2.weight.data.normal_(0, 0.1)  
        self.action.weight.data.normal_(0, 0.1)

    def forward(self, x, boxx, stepx):

        batchsize=x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batchsize,-1)
        
        
        boxx=boxx.view(batchsize,-1)
        boxx=F.relu(self.boxfc1(boxx))
        boxx=F.relu(self.boxfc2(boxx))
        
        stepx=F.relu(self.stepfc(stepx))
        
        x = torch.cat((x, boxx, stepx),1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = F.softmax(self.action(x))

        return action


class Agent(object):

    def __init__(self, weights_path=None):
        
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        
        if weights_path is not None:
            self.eval_net.load_state_dict(torch.load(weights_path))
            self.target_net.load_state_dict(torch.load(weights_path))
        
        #the memory_self is shared by D_short in IL and D_self in RL
        self.memory_long = Memory(p.MEMORY_LONG_CAPACITY) 
        self.memory_self = Memory(p.MEMORY_SELF_CAPACITY)
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=p.LR)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        
    def choose_action(self, x, box, step, valid_mask, greedy=1.0):
        
        x = torch.unsqueeze(torch.FloatTensor(x), 0).cuda()
        box = torch.unsqueeze(torch.FloatTensor(box), 0).cuda()
        step =  torch.unsqueeze(torch.FloatTensor(step), 0).cuda()
        
        if np.random.uniform() < greedy:
            
            actions = self.eval_net.forward(x, box, step)
            actions = actions.cpu()
            
            valid_mask = torch.from_numpy(valid_mask).float()
            valid_mask = valid_mask.unsqueeze(0)
            actions = actions*valid_mask
            
            action = torch.max(actions, 1)[1].data.numpy()
            action = action[0]  

        else:
            
            valid_actions=np.array(valid_mask.nonzero(),dtype=np.int)[0]
            action_index=np.random.randint(0, valid_actions.shape[0])
            action=valid_actions[action_index]
                        
        return action

    
    def learn(self, learning_mode, is_ddqn=True):
    
        if self.learn_step_counter % p.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.learn_step_counter += 1
        
        #only IL by supervised-error only 
        if learning_mode==0:
            samples_from_expert=int(p.BATCH_SIZE*0.5)
        
        #only RL by td-error only
        elif learning_mode==1:
            samples_from_expert=0
        
        #jointly by td-error and supervised error
        elif learning_mode==2:
            samples_from_expert=int(p.BATCH_SIZE*0.5)
        
        #jointly by td-error
        elif learning_mode==3:
            samples_from_expert=int(p.BATCH_SIZE*0.5)
        
        bs1,bbox1,bstep1,ba1,br1,bs_1,bbox_1,bstep_1=self.memory_long.sample(samples_from_expert)
        bs2,bbox2,bstep2,ba2,br2,bs_2,bbox_2,bstep_2=self.memory_self.sample(p.BATCH_SIZE-samples_from_expert)
        
        bs=torch.FloatTensor(np.concatenate((bs1,bs2), axis=0)).cuda()
        bbox=torch.FloatTensor(np.concatenate((bbox1,bbox2), axis=0)).cuda()
        bstep=torch.FloatTensor(np.concatenate((bstep1,bstep2), axis=0)).cuda()
        
        ba=torch.FloatTensor(np.concatenate((ba1,ba2), axis=0)).cuda().long()
        br=torch.FloatTensor(np.concatenate((br1,br2), axis=0)).cuda()

        bs_=torch.FloatTensor(np.concatenate((bs_1,bs_2), axis=0)).cuda()
        bbox_=torch.FloatTensor(np.concatenate((bbox_1,bbox_2), axis=0)).cuda()
        bstep_=torch.FloatTensor(np.concatenate((bstep_1,bstep_2), axis=0)).cuda()
        
        #expert's action's q_value
        q_all_actions = self.eval_net(bs, bbox, bstep)
        q_a_expert = q_all_actions.gather(1, ba)
        
        #margin function
        margin = torch.FloatTensor(np.ones((p.BATCH_SIZE, p.ACTION_NUM))).cuda()*0.8
        t=np.linspace(0,p.BATCH_SIZE-1,p.BATCH_SIZE).astype(np.int)
        margin[t,ba[t,0]]=0
            
        #predicted maximal q_value with margin function
        q_a_predicted = q_all_actions + margin
        q_a_predicted = q_a_predicted.max(1)[0].view(p.BATCH_SIZE, 1)
        
        #compute the supervised loss
        loss1 = torch.mean(q_a_predicted - q_a_expert)
        
        #compute td error by target net
        if not is_ddqn:
            # the loss function of DQN
            q_a_next = self.target_net(bs_, bbox_, bstep_).detach()
            q_a_target = br + p.GAMMA * q_a_next.max(1)[0].view(p.BATCH_SIZE, 1)
            loss2 = self.loss_func(q_a_expert, q_a_target)
        
        else:
            #the loss function of DDQN
            q_a_next = self.target_net(bs_, bbox_, bstep_).detach()
            best_action = self.eval_net(bs_, bbox_, bstep_).max(1)[1].view(p.BATCH_SIZE,1)
            q_a_target=br +  p.GAMMA * q_a_next.gather(1, best_action)
            loss2 = self.loss_func(q_a_expert, q_a_target)

        if learning_mode==0:            
            loss = loss1
        elif learning_mode==1:
            loss = loss2
        elif learning_mode==2:
            loss = loss1 + loss2
        elif learning_mode==3:
            loss = loss2
                    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
