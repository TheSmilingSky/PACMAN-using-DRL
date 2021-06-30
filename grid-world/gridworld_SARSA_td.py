import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

up,right,down,left = 0,1,2,3
grid_size = 10
init_state = (0,0)

class Gridworld:

    def __init__(self,rewards,init_state, terminal_state):
        self.rewards = rewards
        self.init_state = init_state
        self.terminal_state = terminal_state

    def step(self,state,action):
        if action == up:
            next_state = tuple(a - b for a,b in zip(state,(1,0)))
        elif action == down:
            next_state = tuple(a + b for a,b in zip(state,(1,0)))
        elif action == right:
            next_state = tuple(a + b for a,b in zip(state,(0,1)))
        else:
            next_state = tuple(a - b for a,b in zip(state,(0,1)))

        return self.rewards[next_state] , next_state

class Agent:

    def __init__(self,alpha,eps,gamma):
        self.alpha = alpha # hyperparameter
        self.eps = eps # hyperparameter
        self.gamma = gamma #hyperparameter
        self.Q = np.zeros((grid_size,grid_size,4))
        self.V = np.zeros((grid_size,grid_size))
        self.pi = np.ones((grid_size,grid_size,4))
        # self.locked_block = 
        self.initialize_policy()

    def initialize_policy(self):
        self.pi[:,grid_size-1,right] = 0
        self.pi[:,0,left] = 0
        self.pi[0,:,up] = 0
        self.pi[grid_size-1,:,down] = 0

    def update_V(self,state,reward,new_state):
        self.V[state] += self.alpha*(reward + self.gamma*self.V[new_state] - self.V[state])

    def update_pi(self,state):
        self.pi[state] = 0
        self.pi[state,np.flatnonzero(self.Q[state].max() == self.Q[state])] = 1

    def update_Q(self,state,action,reward,new_state,next_action):
        self.Q[state,action] += self.alpha*(reward + self.gamma*self.Q[new_state,next_action] - self.Q[state,action])        

    def get_action(self,state):
        # Epsilon-greedy policy
        if np.random.random() < self.eps: #explore
            if state == (0,0):
                return np.random.choice([right,down])
            elif state == (grid_size-1,0):
                return np.random.choice([right,up])
            elif state == (grid_size-1,grid_size-1):
                return np.random.choice([left,up])
            elif state == (0,grid_size-1):
                return np.random.choice([left,down])
            elif state[0] == 0:
                return np.random.choice([left,down,right])
            elif state[0] == grid_size-1:
                return np.random.choice([left,up,right])
            elif state[1] == grid_size-1:
                return np.random.choice([left,down,up])
            elif state[1] == 0:
                return np.random.choice([right,down,up])  
            else:
                return np.random.randint(4) # np.random.random(x) generates a random number from 0 to n-1 
        else: #exploit
            np.random.choice(np.flatnonzero(self.pi[state] == 1))

# start the grid-world experiment
def experiment(rewards,N_episodes,alpha,eps,gamma):
    env = Gridworld(rewards,(0,0),(5,5))
    agent = Agent(alpha,eps,gamma)
    for episode in range(N_episodes):
        if(episode + 1) % (N_episodes / 100) == 0:
            print("[Experiment {}/{}] ".format(episode + 1, N_episodes))
        state = env.init_state
        action = agent.get_action(state)
        while(state != env.terminal_state):
            reward, new_state = env.step(state,action)
            next_action = agent.get_action(new_state)
            agent.update_V(state,reward,new_state)
            agent.update_Q(state,action,reward,new_state,next_action)
            agent.update_pi(state)
            state, action = new_state, next_action
    print(agent.V)


Erewards = np.zeros((grid_size,grid_size))
Erewards[3,3] = -1
Erewards[4,[5,6]] = -1
Erewards[5,5] = 1
Erewards[5,[6,8]] = -1
Erewards[6,8] = -1
Erewards[7,[3,5,6]] = -1

N_steps = 10000
experiment(Erewards,N_steps,0.15,0.2,0.8)