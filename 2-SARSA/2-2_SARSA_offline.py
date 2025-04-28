#  首先需要有一个Q表，去记录(s, a)对应的Q
#  接着需要去用环境交互去收集经验。
# 需要边收集边去训练Q表。

# 这个环境是一个简单的一维环境：
# 20001000, 这个环境中可以看到一共有8个状态，并且最右面是终态，人物起始在第4位上。
import random
import numpy as np

class MYENV:
    def __init__(self, init_index=4):
        self.env = '20000000'
        self.length = 10
        self.state_now = init_index
        self.state_init = init_index

        self.outa_range_reward = -10
        self.reached_target_reward = 10
        self.has_not_reward = 0
        self.isterminated = False
    
    def step(self, state: int, action: int):
        '''
        输入：state和action

        输出：元组（状态，奖励，是否终止）
        '''

        next_state = state+action
        self.state_now = next_state
        goal_index = 7
        if self.is_outa_range(next_state):
            self.isterminated = True
            return (next_state, self.outa_range_reward, self.isterminated)
        elif next_state == goal_index:
            self.isterminated = True
            return (next_state, self.reached_target_reward, self.isterminated)
        elif state == goal_index:
            self.isterminated = True
            return (next_state, self.has_not_reward, self.isterminated)
        else:
            self.isterminated = False
            return (next_state, self.has_not_reward, self.isterminated)
        
    def reset(self, init_index=4):
        self.__init__(init_index)
    def is_outa_range(self, state):
        return not 0<=state<self.length
    
class QTable:
    def __init__(self, state_size, action_size):
        self.q_table = [[0]*action_size for i in range(state_size)]
        self.state_size = state_size
        self.action_size = action_size
        self.lr = 0.1
        self.gamma = 0.9

    def get_best_action(self, state):
        max_value = max(self.q_table[state])
        indexes = []
        try:
            for i, c in enumerate(self.q_table[state]):
                if c == max_value:
                    indexes.append(i-1)
            
            return random.choice(indexes)
        except:
            print('get_best_actionERROR: ', state)
    
    def get_random_action(self):
        return random.choice([-1,0,1])
    
    def update(self, state, action, nextstate, nextaction,reward):
        action+=1
        nextaction+=1
        self.q_table[state][action] += self.lr*(reward+self.gamma*self.q_table[nextstate][nextaction]-self.q_table[state][action])
    
class Agent:
    def __init__(self, state_size, action_size):
        self.q = QTable(state_size, action_size)
        self.eps = 0.1
    
    def eps_greedy_choose_action(self, state):
        # return self.q.get_random_action()

        if random.random() < self.eps:
            return self.q.get_random_action()
        else:
            return self.q.get_best_action(state)



def train():
    for episode in range(100):
        env.reset(random.choice(range(env.length)))
        # env.reset()
        state = env.state_init
        done = False
        print('episode: ', episode)

        datas = []

        for epoch in range(100):
            env.reset(random.choice(range(env.length)))
            state = env.state_init
            done = False
            
            while not done:
                # print('state:', state)
                action = agent.eps_greedy_choose_action(state)
                # print('action:', action)

                next_state, reward, done = env.step(state, action)
                
                if done:
                    datas.append((state, action, state, action, reward, done))
                else:
                    next_action = agent.eps_greedy_choose_action(next_state)
                    datas.append((state, action, next_state, next_action, reward, done))
                state = next_state

        shuffled_datas = datas[:] # 创建一个副本
        random.shuffle(shuffled_datas)
        for state, action, next_state, next_action, reward, done in shuffled_datas:
            if done:
                agent.q.q_table[state][action+1] += agent.q.lr*(reward-agent.q.q_table[state][action+1])
            else:
                agent.q.update(state, action, next_state, next_action, reward)
def play():
    new_env = MYENV()
    state = new_env.state_now
    done = False
    while not done:
        action = agent.q.get_best_action(state)
        print(state, action)
        state, _, done = new_env.step(state, action)

if __name__ == '__main__':
    env = MYENV()
    agent = Agent(env.length, 3)
    train()
    print(agent.q.q_table)
    print(np.array(agent.q.q_table))
    play()
