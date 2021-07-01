import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import sys
import os

from numpy.random.mtrand import rand

up,right,down,left = 0,1,2,3
grid_size = 10

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
        self.initialize_policy()
        self.initialize_Q()

    def initialize_Q(self):
        self.Q[:,grid_size-1,right] = -sys.maxsize - 1
        self.Q[:,0,left] = -sys.maxsize - 1
        self.Q[0,:,up] = -sys.maxsize - 1
        self.Q[grid_size-1,:,down] = -sys.maxsize - 1
        #These initialization are to make those blocked cells in the grid-world
        self.Q[1,1:5,down] = -sys.maxsize - 1
        self.Q[3,1:4,up] = -sys.maxsize - 1
        self.Q[2,1,right] = -sys.maxsize - 1
        self.Q[3:8,3,right] = -sys.maxsize - 1
        self.Q[8,4,up] = -sys.maxsize - 1
        self.Q[2:8,5,left] = -sys.maxsize - 1
        self.Q[1,6:9,down] = -sys.maxsize - 1
        self.Q[3,6:9,up] = -sys.maxsize - 1
        self.Q[2,5,right] = -sys.maxsize - 1
        self.Q[2,9,left] = -sys.maxsize - 1

    def initialize_policy(self):
        self.pi[:,grid_size-1,right] = 0
        self.pi[:,0,left] = 0
        self.pi[0,:,up] = 0
        self.pi[grid_size-1,:,down] = 0
        #These initialization are to make those blocked cells in the grid-world
        self.pi[1,1:5,down] = 0
        self.pi[3,1:4,up] = 0
        self.pi[2,1,right] = 0
        self.pi[3:8,3,right] = 0
        self.pi[8,4,up] = 0
        self.pi[2:8,5,left] = 0
        self.pi[1,6:9,down] = 0
        self.pi[3,6:9,up] = 0
        self.pi[2,5,right] = 0
        self.pi[2,9,left] = 0
        # print(self.pi[0,0])

    def update_V(self,state,reward,new_state):
        self.V[state] = self.V[state] + self.alpha*(reward + self.gamma*self.V[new_state] - self.V[state])

    def update_pi(self,state):
        self.pi[state] = 0
        self.pi[state][np.flatnonzero(self.Q[state] == self.Q[state].max())] = 1

    def update_Q(self,state,action,reward,new_state,next_action):
        # print(action.shape)
        self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*self.Q[new_state][next_action] - self.Q[state][action])        

    def get_action(self,state):
        # Epsilon-greedy policy
        if np.random.random() < self.eps: #explore
            # print('explore')
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
                # for blocking
                if state[0] == 2:
                    return np.random.choice([up,down])
                else:
                    return np.random.choice([left,down,up])
            elif state[1] == 0:
                # for blocking
                if state[0] == 2:
                    return np.random.choice([down,up])
                else:
                    return np.random.choice([right,down,up])
            #These are to make those blocked cells in the grid-world
            elif state[0] == 1 and state[1] in range(1,5):
                return np.random.choice([right,left,up])
            elif state[0] == 3 and state[1] in range(1,3):
                return np.random.choice([right,left,down])
            elif state == (3,3):
                return np.random.choice([left,down])
            elif state[0] in range (4,8) and state[1] == 3:
                return np.random.choice([down,left,up])
            elif state == (8,4):
                return np.random.choice([right,left,down])
            elif state == (2,5):
                return np.random.choice([down,up])
            elif state[0] == 1 and state[1] in range(6,9):
                return np.random.choice([right,left,up])
            elif state[0] == 3 and state[1] in range(6,9):
                return np.random.choice([right,left,down])
            else:
                return np.random.randint(4) # np.random.random(x) generates a random number from 0 to n-1 
        else: #exploit
            # print('exploit')
            # print(self.pi[state])
            return np.random.choice(np.flatnonzero(self.pi[state] == 1))

def plot_arrow(data):
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            # print(data[0,0])
            if data[i,j,up] == 1:
                plt.arrow(j,i+0.25,0,-0.5,width = 0.05)
            if data[i,j,right] == 1:   
                plt.arrow(j-0.25,i,0.5,0,width = 0.05)
            if data[i,j,down] == 1:
                plt.arrow(j,i-0.25,0,0.5,width = 0.05)
            if data[i,j,left] == 1:
                plt.arrow(j+0.25,i,-0.5,0,width = 0.05)

def plot(data,policy,state):
    # create discrete colormap
    _ , ax = plt.subplots()
    ax.imshow(data)
    plot_arrow(policy)
    Drawing_colored_circle = plt.Circle((state[1],state[0]),0.25)
    ax.add_artist( Drawing_colored_circle )

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-0.5, 10, 1));
    ax.set_yticks(np.arange(-0.5, 10, 1));
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.savefig(os.path.join(output_dir, "grid_world_SARSA.jpg"), bbox_inches="tight")
    plt.show()

# start the grid-world experiment
def experiment(rewards,N_episodes,alpha,eps,gamma):
    env = Gridworld(rewards,(0,0),(5,5))
    agent = Agent(alpha,eps,gamma)
    for episode in range(N_episodes):
        if(episode + 1) % (N_episodes / 1000) == 0:
            print("[Experiment {}/{}] ".format(episode + 1, N_episodes))
        state = env.init_state
        action = agent.get_action(state)
        # print(state,action)
        # for x in range(1000):
        while(state != env.terminal_state):
            reward, new_state = env.step(state,action)
            next_action = agent.get_action(new_state)
            agent.update_V(state,reward,new_state)
            agent.update_Q(state,action,reward,new_state,next_action)
            agent.update_pi(state)
            state, action = new_state, next_action
        print(state,action)
    plot(agent.V,agent.pi,state)


Erewards = np.zeros((grid_size,grid_size))
Erewards[3,3] = -1
Erewards[4,[5,6]] = -1
Erewards[5,5] = 1
Erewards[5,[6,8]] = -1
Erewards[6,8] = -1
Erewards[7,[3,5,6]] = -1

N_steps = 10000
experiment(Erewards,N_steps,0.15,0.5,0.8)