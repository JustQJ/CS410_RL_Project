import sys,os
from env.chooseenv import make
from submission import *
import torch
import matplotlib.pyplot as plt
plotx = [] #record the process 
ploty =[]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)


def test(model,max_score,episode):
    env=make("snakes_3v3",conf=None)
    score=0
    for epoch in range(10):
        state=env.reset()
        while True:
            actions=[]
            for agent in range(6):
                direction,feature=resnet_direction_and_obs(state[agent],state[agent]["controlled_snake_index"])
                out=model.get_action(np.array(feature),mode=False)
                action=net_action2real_action(out,direction)
                action=check_and_replace(state[agent],action)
                actions.append(action)
            next_state,reward,done,_,info=env.step(env.encode(actions))
            state=next_state

            if done:
                obs=next_state[0]
                for agent_i in {2,3,4,5,6,7}:
                    score+=len(obs[agent_i])
                break

    score=score/60
    if score>max_score or (episode%1000==0) or score>12:
        filepath=os.path.join('checkpoint_model_episode_{}_score_{}.pth'.format(episode,score))  #最终参数模型
        model.save_parameters(filepath)
    return score


def main():
    set_seed(2)
    print("start to train")
    env=make("snakes_3v3",conf=None)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("we are using:",device)
    model=RLAgent(act_dim=3)
    #actor_net = os.path.dirname(os.path.abspath(__file__)) +"/checkpoint_model_episode_1050_score_12.816666666666666.pth"
    #model.load_parameters(actor_net)
    episode=0
    Max_score=-float("inf")
    totalstep=0
    while episode<2000:
        state=env.reset()
        episode+=1
        while True:
            action_list,logits_list,obs_list,next_obs_list,Done=[],[],[],[],[]
            for agent in range(6):
                direction,obs=resnet_direction_and_obs(state[agent],
                                                           state[agent]["controlled_snake_index"])
                output=model.get_action(np.array(obs),mode=True)
                action=net_action2real_action(output,direction)
                action=check_and_replace(state[agent],action)

                logits_list.append(output)
                action_list.append(action)
                obs_list.append(obs)
            next_state,reward,done,_,info=env.step(env.encode(action_list))

            for agent in range(6):
                direction,next_obs=resnet_direction_and_obs(next_state[agent],
                                                           next_state[agent]["controlled_snake_index"])
                next_obs_list.append(next_obs)
                Done.append(check_death(state[agent],next_state[agent]))

            for agent in range(6):
                model.replay_buffer.push(obs_list[agent],logits_list[agent],reward[agent],next_obs_list[agent],
                                         Done[agent])

            if totalstep*6>model.replay_buffer.max_size:
                model.train()
            state=next_state
            totalstep+=1
            if done:
                print("Episode:%d"%(episode))
                if (episode%50)==0:
                    score=test(model,Max_score,episode)
                    Max_score=max(Max_score,score)
                    print("Episode:%d,score:%f"%(episode,score))
                    plotx.append(episode)
                    ploty.append(score)
                break
    print("finish")


if __name__=='__main__':
    main()
    plt.plot(plotx,ploty)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()