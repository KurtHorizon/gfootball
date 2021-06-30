import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import gfootball.env as football_env
import warnings
warnings.filterwarnings('ignore')

# Hyper Parameters
STATE_DIM = 115
ACTION_DIM = 19
STEP = 100000    
SAMPLE_NUMS = 200      
_gamma=0.993
_lr=0.00005
max_grad_norm=0.5
ruleBase=False  #True = ruleBase agent, False = RL agent
herit=True
save_net=True
print_info=False
test_ = False
render_=False
writevideo_=False
dump_=False

total_rew=0.0
total_step=0

class ActorNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out

class ValueNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def roll_out(actor_network,env,sample_nums,value_network,init_state):
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state
    step_num=0
    for j in range(sample_nums):
        states.append(state)
        log_softmax_action = actor_network(Variable(torch.Tensor([state])))
        softmax_action = torch.exp(log_softmax_action)
        action=0
        if ruleBase==True:
            action=rule(step_num)
        if action==0:
            action = np.random.choice(ACTION_DIM,p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state,reward,done,_ = env.step(action)
      
        if print_info == True:
            if reward > 0 or j%15==0:
                print('itr: ' + str(j) + ', action=' + str(action) + ', reward=' + str(reward))
                if reward==1:
                    print("goal!\n")
        global total_rew
        global total_step
        if reward==1:
            total_rew=total_rew+1
            total_step=total_step+step_num
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        step_num=step_num+1
        if done:
            is_done = True
            state = env.reset()
            break
    if not is_done:
        final_r = value_network(Variable(torch.Tensor([final_state]))).cpu().data.numpy()

    return states,actions,rewards,final_r,state

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
    
  #ruleBase: pass and shot immediately  
def rule(step):
    if step<=1:
        return 9
    if  step <=10:
        return 12
    return 0

def main():
    # load network
    if herit == False:
        value_network = ValueNetwork(input_size = STATE_DIM,hidden_size = 40,output_size = 1)
        actor_network = ActorNetwork(STATE_DIM,128,ACTION_DIM)
    else:
        value_network  = torch.load('../param/net_param/critic_net.pkl')
        actor_network  = torch.load('../param/net_param/actor_net.pkl')
    #init optim
    actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr =_lr)
    value_network_optim = torch.optim.Adam(value_network.parameters(),lr=_lr)

    steps =[]
    test_results =[]
    env = football_env.create_environment(
        env_name='academy_3_vs_1_with_keeper', 
        representation='simple115', 
        render=render_)
        #write_video=writevideo_, 
        #write_goal_dumps=dump_,
        #logdir='../video/v2.avi'  )
    init_state = env.reset()
    for step in range(STEP):
        
        states,actions,rewards,final_r,current_state = roll_out(actor_network,env,SAMPLE_NUMS,value_network,init_state)
        init_state = current_state
        actions_ten=torch.Tensor(actions)
        actions_var = Variable(actions_ten.view(-1,ACTION_DIM))  #actions actually happen
        states_ten=torch.Tensor(states)
        states_var = Variable(states_ten.view(-1,STATE_DIM))  #state actually happen

        # train actor network: 
        actor_network_optim.zero_grad()
        log_softmax_actions = actor_network(states_var)
        vs = value_network(states_var).detach()
        qs = Variable(torch.Tensor(discount_reward(rewards,_gamma,final_r)))
        advantages = qs - vs
        #policy gradient: loss= -log(prob)*vt
        actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(),max_grad_norm)
        actor_network_optim.step()

        #train value network: MSEloss
        value_network_optim.zero_grad()
        target_values = qs
        values = value_network(states_var)   
        criterion = nn.MSELoss()
        value_network_loss = criterion(values,target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(),max_grad_norm)
        value_network_optim.step()

        if print_info==1:
            print("Successfilly updated")
        if (step+1)%50==0:
            env=football_env.create_environment(env_name = 'academy_3_vs_1_with_keeper', representation='simple115',render=render_)
            global total_rew
            global total_step
            print(total_rew)
            print(total_step)
            if save_net == True and ruleBase==False:
                    torch.save(actor_network, '../param/net_param/actor_net.pkl')
                    torch.save(value_network, '../param/net_param/critic_net.pkl')

if __name__ == '__main__':
    main()
    