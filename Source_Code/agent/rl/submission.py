import torch
import os
import sys
from pathlib import Path
base_dir=Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from Resnet import *
Replay_buffer_size=30000
Batch_size=256
View_size=10

#priority relay buffer
class SumTree:
    def __init__(self,buffer_size):
        self.data_pointer = 0
        self.size = buffer_size
        self.sum_tree = np.zeros(2*buffer_size-1) #二叉树存储优先级
        self.transitions = np.zeros(buffer_size,dtype=object) #存储状态

    def add_transition(self,priority,transition): #添加新的数据
        treeind = self.data_pointer+self.size - 1 #计算在二叉树中的位置
        self.transitions[self.data_pointer] = transition #存入数据
        self.update(treeind,priority) #更新二叉树

        self.data_pointer = (self.data_pointer+1) % self.size

    def update(self,treeind,priority):
        change = priority-self.sum_tree[treeind]
        self.sum_tree[treeind]=priority
        while treeind!=0:    #回溯更新整颗树
            treeind = (treeind-1)//2
            self.sum_tree[treeind] += change

    def get_transition(self,priority):

        #从头节点遍历到叶子节点，取回叶子节点的数据
        parent = 0
        while True:
            leftchild = 2*parent+1
            rightchild = leftchild+1
            if leftchild>=len(self.sum_tree):
                tree_ind = parent
                break
            else:
                if priority<=self.sum_tree[leftchild]:
                    parent = leftchild
                else:
                    priority = priority - self.sum_tree[leftchild]
                    parent = rightchild
        transition_ind = tree_ind - self.size+1

        return tree_ind, self.sum_tree[tree_ind], self.transitions[transition_ind]

    def root(self):
        return self.sum_tree[0] #总的priority


#在replay-buffer中放入记录
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.sumTree = SumTree(buffer_size)
        self.max_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = 0.01 
        self.alpha = 0.3
        self.beta = 0.2
        self.beta_incrasing = 0.0002
        self.abs_err_upper =1

    def push(self, state, logits, reward, next_state, done):
        transition = (state, logits, reward, next_state, done)
        priority = np.max(self.sumTree.sum_tree[-self.max_size:]) #从叶子节点中选择最大的优先级
        if priority == 0:
            priority = self.abs_err_upper    
        self.sumTree.add_transition(priority,transition) #添加到buffer中去

    def get_batches(self):

        sample_ind = np.empty((self.batch_size,), dtype=np.int32)
        sample_batch = np.zeros(self.batch_size,dtype=object) #存储状态
        weight = np.empty((self.batch_size,1))

        segment = self.sumTree.root()/self.batch_size

        self.beta = np.min([1.,self.beta+self.beta_incrasing]) #增加beta
        min_p = np.min(self.sumTree.sum_tree[-self.max_size:])/self.sumTree.root()

        for i in range(self.batch_size):
            l,r = segment*i,segment*(i+1)
            pri = np.random.uniform(l,r)
            tree_ind,priority,transition = self.sumTree.get_transition(pri)

            probability = priority/self.sumTree.root()
            weight[i,0] = np.power(probability/min_p,-self.beta)
            sample_ind[i] = tree_ind
            sample_batch[i] = transition
        
        return sample_ind, sample_batch, weight

    def batch_update(self, tree_indexs, abs_error):
        abs_error = abs_error+self.epsilon #避免为0
        abs_error = np.minimum(abs_error,self.abs_err_upper)
        prioritys = np.power(abs_error,self.alpha)
        for tree, pr in zip(tree_indexs, prioritys):
            self.sumTree.update(tree, pr)

    def __len__(self):
        return len(self.replay_buffer)


def net_action2real_action(output,direction):
    Naction2Raction_list={"up":[2,1,3],"right":[1,3,0],"down":[3,0,2],"left":[0,2,1]}
    tmp=Naction2Raction_list[direction]
    return tmp[output]


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y,weight):
        return torch.mean(weight*torch.pow((x - y), 2))


#mode表示是否为训练模式,true表示为训练模式，false表示为测试模式
class RLAgent(object):
    def __init__(self, act_dim):

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.replay_buffer=ReplayBuffer(Replay_buffer_size,Batch_size)
        self.epsilon=1
        self.act_dim=act_dim
        self.eval_network=ResNet18(11,self.act_dim).to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.target_network=ResNet18(11,self.act_dim).to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_function=My_loss().to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer=torch.optim.Adam(self.eval_network.parameters(),lr=1e-5)

    def get_action(self,obs,mode):
        obs=torch.from_numpy(obs).view(-1,11,10,10).float().to(self.device)
        if mode:  #带有探索的行为
            next_state_q=self.target_network(obs).detach()
            if self.epsilon>0.05:
                self.epsilon=self.epsilon-0.00002
            if random.random()<=self.epsilon:
                return random.randint(0,self.act_dim-1)
            else:
                action=np.argmax(next_state_q.cpu().numpy())
                return action
        else:
            next_state_q=self.target_network(obs).detach()
            action=np.argmax(next_state_q.cpu())
            return action

    def train(self):
        sample_ind, sample_batch, weight = self.replay_buffer.get_batches()

        state_batch=torch.FloatTensor(np.array([_[0] for _ in sample_batch])).to(self.device)
        action_batch=torch.LongTensor(np.array([_[1] for _ in sample_batch])).reshape(Batch_size,1).to(self.device)
        reward_batch=torch.Tensor(np.array([_[2] for _ in sample_batch])).reshape(Batch_size,1).to(self.device)
        next_state_batch=torch.FloatTensor(np.array([_[3] for _ in sample_batch])).to(self.device)
        done=torch.Tensor(np.array([_[4] for _ in sample_batch])).view(Batch_size,1).to(self.device)
        weight=torch.tensor(weight).view(Batch_size,1).to(self.device)
        Q_eval=self.eval_network(state_batch).gather(1,action_batch)
        Q_next=self.target_network(next_state_batch).detach()
        Q_next_eval = self.eval_network(next_state_batch)
        max_action_indx = Q_next_eval.max(1)[1]
        indexrow = np.array(list(range(256)))
        Q_target = reward_batch+(1-done)*0.99*Q_next[indexrow,max_action_indx].view(Batch_size,1)
        Qval = Q_eval.cpu().detach().numpy()
        Qtar = Q_target.cpu().numpy()
        abs_errs = np.abs(Qval,Qtar)
        self.replay_buffer.batch_update(sample_ind,abs_errs)
        loss=self.loss_function(Q_eval,Q_target,weight)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_copy()
        return loss

    def soft_copy(self):
        for target_param, param in zip(self.target_network.parameters(), self.eval_network.parameters()):
            target_param.data.copy_(target_param.data * 0.98 + param.data*0.02)

    def save_parameters(self,save_path):
        torch.save(self.target_network.state_dict(),save_path)

    def load_parameters(self,load_path):
        tmp=torch.load(load_path,map_location=self.device)
        self.eval_network.load_state_dict(tmp)
        self.target_network.load_state_dict(tmp)


def check_and_replace(state,action):
    snake_position=[]
    height=state["board_height"]
    width=state["board_width"]
    snake_idx=state["controlled_snake_index"]
    my_head=state[snake_idx][0]
    for index in state.keys():
        if index==1:
            pass
        elif isinstance(index,int):
            snake=state[index]
            for body in snake[:-1]:
                snake_position.append(body)

    head_0=[(my_head[0]+height-1)%height,my_head[1]]
    head_1=[(my_head[0]+height+1)%height,my_head[1]]
    head_2=[my_head[0],(my_head[1]+width-1)%width]
    head_3=[my_head[0],(my_head[1]+width+1)%width]
    tmp={0:head_0,1:head_1,2:head_2,3:head_3}
    next_head=tmp[action]
    if next_head in snake_position:
        for next_action,next_position in tmp.items():
            if next_position in snake_position:
                pass
            else:
                action=next_action
                break
    return action


def check_death(state,next_state):
    idx=state["controlled_snake_index"]
    cur_len=len(state[idx])
    next_len=len(next_state[idx])
    if next_len<cur_len:
        return 1
    else:
        return 0


#编码action为one-hot向量
def to_joint_action(action):
    joint_action_ = []
    action_a = action
    each = [0] * 4
    each[action_a] = 1
    joint_action_.append(each)
    return joint_action_


#提交的时候移除下面的注释
model =RLAgent(act_dim=3)
actor_net = os.path.dirname(os.path.abspath(__file__)) +"/checkpoint_model_episode_300_score_13.633333333333333.pth"
model.load_parameters(actor_net)


def my_controller(observation_list, action_space_list, is_act_continuous):
    obs=observation_list.copy()
    direction,feature=resnet_direction_and_obs(obs,obs["controlled_snake_index"])
    out=model.get_action(np.array(feature),mode=False)
    action=net_action2real_action(out,direction)
    action=check_and_replace(obs,action)
    action=to_joint_action(action)
    return action
