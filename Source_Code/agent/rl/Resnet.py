import torch
from torch import nn
import numpy as np
import random
from torch.nn import functional as F
View_size=10


def resnet_direction_and_obs(state,agent_idx):
    snake=state[agent_idx]
    head=snake[0]
    body=snake[1]
    height=state["board_height"]
    width=state["board_width"]
    direction=None
    if (head[0]==((body[0]+1)%height)) and (head[1]==body[1]):
        direction="up"
    elif (head[0]==((body[0]+height-1)%height)) and (head[1]==body[1]):
        direction="down"
    elif (head[0]==body[0]) and (head[1]==((body[1]+1)%width)):
        direction="right"
    elif (head[0]==body[0]) and (head[1]==((body[1]+width-1)%width)):
        direction="left"

    full_map=np.zeros((11,height,width))
    view_map=np.zeros((11,View_size,View_size))
    #layer0:所有snake的位置：
    for index in state.keys():
        if index==1:
            continue
        elif isinstance(index,int):
            data=state[index]
            for item in data:
                full_map[0,item[0],item[1]]=1

    #layer1:控制的蛇头的位置：
    data=state[agent_idx]
    head=data[0]
    full_map[1,head[0],head[1]]=1

    #layer2:我方所有的蛇头的位置
    #先获取阵营：
    if agent_idx<=4:
        teammates=[2,3,4]
        enemies=[5,6,7]
    else:
        teammates=[5,6,7]
        enemies=[2,3,4]

    for teammate in teammates:
        teammate_position=state[teammate]
        teammate_head=teammate_position[0]
        full_map[2,teammate_head[0],teammate_head[1]]=1

    #layer3:获取敌人的蛇头的位置
    for enemy in enemies:
        enemy_position=state[enemy]
        enemy_head=enemy_position[0]
        full_map[3,enemy_head[0],enemy_head[1]]=1

    #layer4:豆的位置：
    beam_positions=state[1]
    for item in beam_positions:
        full_map[4,item[0],item[1]]=1

    #layer5:控制的蛇的全部位置：
    data=state[agent_idx]
    for item in data:
        full_map[5,item[0],item[1]]=1

    #layer6-7:队友的蛇的全身位置
    layer=6
    for teammate in teammates:
        if teammate==agent_idx:
            continue
        else:
            teammate_position=state[teammate]
            for position in teammate_position:
                full_map[layer,position[0],position[1]]=1
            layer+=1

    #layer8-10:敌人的蛇的全身位置：
    layer=8
    for enemy in enemies:
        enemy_position=state[enemy]
        for position in enemy_position:
            full_map[layer,position[0],position[1]]=1
        layer+=1

    #开始设置转换图
    snake=state[agent_idx]
    head=snake[0]
    for z in range(11):
        for x in range(View_size):
            for y in range(View_size):
                dx=(head[0]+height+(x-View_size/2))%height
                dy=(head[1]+width+(y-View_size/2))%width
                dx,dy=int(dx),int(dy)
                view_map[z,x,y]=full_map[z,dx,dy]

    #旋转图纸
    view_map=resnet_transform(view_map,direction)
    return direction,view_map


#经过变换，使得头始终朝上，这样可以选择的行为变成三个
def resnet_transform(feature, direction):
    result=None
    if direction == "up":
        result = feature
    elif direction == "right":
        # 逆时针旋转90度
        result=np.rot90(feature, 1,axes=(1,2))
    elif direction == "down":
        # 逆时针旋转180度
        result=np.rot90(feature,2,axes=(1,2))
    elif direction == "left":
        # 逆时针旋转270度
        result=np.rot90(feature,3,axes=(1,2))
    return result


class ResBlk(nn.Module):

    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=1,stride=1,padding=0)
        self.bn2=nn.BatchNorm2d(ch_out)

        self.extra=nn.Sequential()
        if ch_out!=ch_in:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):

        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out=self.extra(x)+out
        out=F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self,input_channel,output_dim):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )

        self.blk1 = ResBlk(32, 64, stride=1)
        self.layer1=nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU()
        )
        self.layer2=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU()
        )
        self.layer3=nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU()
        )
        #定义A网络
        self.layer4_a=nn.Sequential(
            nn.Linear(128,output_dim),
        )
        #定义V网络
        self.layer4_v=nn.Sequential(
            nn.Linear(128,1),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = F.adaptive_avg_pool2d(x, [2,2])
        x = x.view(x.size(0), -1)
        x_1=self.layer1(x)
        x_2=self.layer2(x_1)
        x_3=self.layer3(x_2)
        a=self.layer4_a(x_3)
        v=self.layer4_v(x_3)
        x_4=v+(a-torch.mean(a))
        return x_4


