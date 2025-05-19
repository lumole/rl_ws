'''
v_criterion = torch.nn.MSELoss()
main_actor_optimizer = torch.optim.Adam(main_actor.parameters(), lr = lr)
main_critic_optimizer = torch.optim.Adam(main_critic.parameters(), lr = lr)#应该去更新主网络
主循环：
    每个episode: 
        从main_actor根据当前s_t得到a_t#要使用main_actor获取
        给a_t加上随机噪声，保证探索性
        把a_t与环境交互，得到一个转移元组(s_t, a_t, r_t, s_{t+1}, done)

        # 开始更新
        先从replay buffer中获取一组点 : states, actions, rewards, nextstates, dones = minibatch
        
        ##计算actor的loss
        L_actor = - main_Q(states, main_actor(states)).mean()#更新actor的时候全都是main网络
        main_actor_optimizer.zero_grad()
        L_actor.backward()
        main_actor_optimizer.step()

        ##计算critic的loss
        with torch.no_grad():
            y = rewards + gamma*(1-dones)*target_Q(nextstates, target_actor(nextstates))#只有更新critic的时候才是target
        L_Q = v_criterion(main_Q(states,actions), y)
        main_critic_optimizer.zero_grad()
        L_Q.backward()
        main_critic_optimizer.step()

        软更新target_actor和target_Q，更新率tau是0.001#论文中建议每一步都软更新
'''

