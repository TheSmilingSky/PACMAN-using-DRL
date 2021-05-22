# PAC MAN

## Reinforcement Learning Basics

Learning of any form is based on the idea of interaction with the environment to achieve goals.

We would learn a computational approach to learning from interaction.

It includes: 

1. Sensing the environment
2. Taking actions to affect the state.
3. Goal relating to the state of the environment 

### Elements of Reinforcement learning:

* **Policy** : A policy defines the learning agent’s way of behaving at a given time. A  policy is a mapping from perceived states of the environment to actions to be taken when in those states. Policies may be stochastic.

* **Reward** A reward signal defines the goal in a reinforcement learning problem. On each time step, the environment sends to the reinforcement learning agent a single number, a reward. The agent’s sole objective is to maximize the total reward it receives over the long run.

  The reward sent to the agent at any time depends on the agent’s current action and the current state of the agent’s environment. The agent cannot alter the process that does this. The only way the agent can influence the reward signal is through its actions, which can have a direct effect on reward, or an indirect effect through changing the environment’s state.

* **Value** : Whereas the reward signal indicates what is good in an immediate sense,a value function specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Whereas rewards determine the immediate, intrinsic desirability of environmental states, values indicate the long-term desirability of states after taking into account the states that are likely to follow, and the rewards available in those states

* **Model** : Model mimics the behaviour of the environment, or more generally, that allows inferences to be made about how the environment will behave.

When we say that a reinforcement learning agent’s goal is to maximize a numerical reward signal, we of course are not insisting that the agent has to
actually achieve the goal of maximum reward. Trying to maximize a quantity
does not mean that that quantity is ever maximized. The point is that a reinforcement learning agent is always trying to increase the amount of reward it receives. Many factors can prevent it from achieving the maximum, even if one exists.

### Reinforcement Learning in Tic Tac Toe

* Tic-Tac-Toe would be approached making use of a value function.
* Table of number(state's value) one for each possible state of the game -  the whole table is the learned value function.

## Multi-armed bandits

* Purely evaluative feedback indicates how good the action taken was, but not whether it was the best or the worst action possible.
* Purely instructive feedback, on the other hand, indicates the correct action to take, independently of the action actually taken.

We are not gonna consider either purely evaluative or purely instruction feedback, we would deal with a mix of these.

In our k -armed bandit problem, each of the k actions has an expected or mean reward
given that that action is selected; let us call this the value of that action. We denote the
action selected on time step t as $A_t$ , and the corresponding reward as $R_t$ . The value then
of an arbitrary action $a$, denoted $q_{*} (a)$, is the expected reward given that a is selected:
$$
q_{*} (a) = E[R_t | A_t  = a]
$$
The  estimated value of action a at the time step t is $Q_t(a)$ 

* **Action Selection Rule** - The simplest action selection rule is to select one of the actions with the highest estimated value.(This is not expected value  $q_{*} (a)$ but is instead the estimated value $Q_t(a)$ )

