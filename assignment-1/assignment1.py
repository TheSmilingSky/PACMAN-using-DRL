import os
import numpy as np
import matplotlib.pyplot as plt

class Environment:

    def __init__(self,probs):
        self.probs = probs # Success probability for each arm (A part of Environment)

    def step(self, action):
        return 1 if (np.random.random() < self.probs[action]) else 0

class Agent:

    def __init__(self, nActions, eps):
        self.nActions = nActions
        self.eps = eps
        self.n = np.zeros(nActions, dtype=np.int) # action counts n(a)
        self.Q = np.zeros(nActions, dtype=np.float) # value Q(a)

    def update_Q(self, action, reward):
        # Update Q_action value given (action, reward)
        self.n[action] += 1
        self.Q[action] += (1.0/self.n[action]) * (reward - self.Q[action])

    def get_action(self):
        # Epsilon-greedy policy
        if np.random.random() < self.eps: #explore
            return np.random.randint(self.nActions) # np.random.random(x) generates a random number from 0 to n-1 
        else: #exploit
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max())) 
            # self.Q == self.Q.max() would generate a numpy array with 0 value where Q != Q.max()
            # faltnonzero returns an array of indices that are non zero
            # choice chooses a random element from the elements of a given array

#start multi-armed bandit experiment
def experiment(probs, N_episodes, eps):
    env = Environment(probs) #initialize arm probabilities
    agent = Agent(len(env.probs), eps)
    regrets = []
    for episode in range(N_episodes):
        action = agent.get_action()
        reward = env.step(action)
        agent.update_Q(action, reward)
        regret = agent.Q.max() - reward
        regrets.append(regret)
    return np.array(regrets)

# Settings
probs = [0.10, 0.50, 0.60, 0.80, 0.10,
         0.25, 0.60, 0.45, 0.75, 0.65] # bandit arm probabilities of success
N_experiments = 10000 # number of experiments to perform
N_steps = 500 # number of steps (episodes)
eps = [0.001,0.01,0.1,0.9]  # probability of random exploration (fraction)
save_fig = True # save file in same directory
output_dir = os.path.join(os.getcwd(), "output_assignment1")

# Run multi-armed bandit experiments
print("Running multi-armed bandits with nActions = {}, eps = {}".format(len(probs), eps))
R = np.zeros((N_steps,len(eps)))  # regret history sum
for j in range(len(eps)):
    for i in range(N_experiments):
        regrets = experiment(probs, N_steps, eps[j])
        if (i + 1) % (N_experiments / 100) == 0:
            print("[Experiment {}/{}] ".format(i + 1, N_experiments) +
                "eps = {}, ".format(eps[j]) +
                "regret_avg = {}".format(np.sum(regrets) / len(regrets)))
        R[:,j] += regrets

# Plot regret results
for i in range(len(eps)):
    R_pct =  R[:,i] / np.float(N_experiments)
    steps = list(np.array(range(len(R_pct)))+1)
    plt.plot(steps, R_pct, ".",
            linewidth=4,
            label="eps : {}".format(eps[i]))

plt.xlabel("Step")
plt.ylabel("Average Regret")
leg = plt.legend(loc='upper left', shadow=True)
plt.grid()
ax = plt.gca()
plt.xlim([1, N_steps])
if save_fig:
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "regrets_multiple_eps.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()
