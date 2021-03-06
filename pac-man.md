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

## Finite Markov Decision Process









### Policies and Value function

* In an MDP we have two value functions - one is the value function $v_{\pi}(s)$  and other is the action-value function $q_{\pi}(s,a)$. These functions estimate how good it is for the agent to be in a given state or how good it is to perform a given action in a given state respectively.

* "how good" here is defined quantitatively in terms of the expected return(sum of all future rewards with discounts)

* These value functions are defined with respect to particular ways of acting, called policies.

  [Ex 3.11](https://blog.csdn.net/ballade2012/article/details/90552538)

* The value function of a state a under a policy $\pi$, denoted $v_{\pi}(s)$, is the expected return when starting in s and following $\pi$ thereafter.	

* Similarly, we define the value of taking action a in state s under a policy $\pi$, denoted $q_{\pi}(s,a$ )
  as the expected return starting from s, taking the action a, and thereafter following policy $\pi$ 

* The value functions satisfies a recursive relationship.

* This recursive relation is called the Bellman equation.  Look at the Bellman diagram to understand the Bellman equation. The Bellman equation averages over all the possibilities, weighting each by its probability of occurring. It states that the value of the start state must equal the value of the expected next state, plus the reward expected along the way.

  [Ex 3.17]()

  [Ex 3.18]()

  [Ex3.19]()

### Optimal Policies and Optimal Value Functions

* A reinforcement learning task roughly means finding a policy that achieves a lot of reward over the long run. 
* So for a finite MDPs a policy is said to be an optimal policy if its expected return is greater than or equal to all other policies.
* There could be more than one optimal policies but we denote all the optimal policies by $\pi_{*}$ 
* Optimal policies share same state value function and state-action value function.
* Read Eq(3.19) and Eq(3.20) for getting Bellman equation for optimal values. 

* Once we have the $V_{*}$ for each of these states we can easily find out the optimal policy in a one-step search  because the policy which would be greedy with respect to the optimal evaluation function $v_{*}$ is an optimal policy. A greedy policy would be actually optimal in the long-term sense because $v_{*}$ already takes into account the reward consequences of all possible future behaviour. 
* By means of $v_{*}$ , the optimal expected long-term return is turned into a quantity that is locally and immediately available for each state. Hence, a one-step-ahead search yields the long-term optimal actions.

* This solution to a Reinforcement learning problem relies on three assumptions
  1. We accurately know the dynamics of the environment
  2. We have enough computational resources to complete the computation of the solution
  3. The Markov property

## Summary

* Reinforcement learning is about learning from interaction
  how to behave in order to achieve a goal.
* The reinforcement learning agent and its environment interact over a sequence of discrete time steps.
* The specification of their interface defines a particular task: the actions are the choices made by the agent; the states are the basis for making the choices; and the rewards are the basis for evaluating the choices.
* Everything inside the agent is completely known and controllable by the
  agent; everything outside is incompletely controllable but may or may not be completely
  known.
* A policy is a stochastic rule by which the agent selects actions as a function of states. The agent’s objective is to maximize the amount of reward it receives over time.
* The undiscounted formulation is appropriate for episodic tasks, in which the agent–environment interaction breaks naturally into episodes; the discounted formulation is appropriate for continuing tasks, in which the interaction does not naturally break into episodes but continues without limit.
  We try to define the returns for the two kinds of tasks such that one set of equations can apply to both the episodic and continuing cases.

* In reinforcement learning we are very much concerned with cases in
  which optimal solutions cannot be found but must be approximated in some way.

## Temporal-Difference Learning

* Like Dynamic Programming, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome.

* Monte Carlo methods wait until the return following the visit is known, then use that return as a target for $V(S_t)$
  $$
  V(S_t) <- V(S_t) + \alpha [G_t - V(S_t)]
  $$
  where $G_T$ is the actual return following time t and $\alpha$ is a constant step-size parameter.

* Whereas Monte Carlo methods must wait until end of the episode to determine the increment to V(S_t), TD methods wait only until the next time step.

$$
V(S_t) < - V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$

This is TD(0) or one-step TD, because it is a special case of the TD($\lambda$)

* If the array V does not change during the episode (as it does not in Monte Carlo methods), then the Monte Carlo error can be written as a sum of TD errors.
  $$
  G_t - V(S_t) = \sum^{T-1}_{k=t} \gamma^{k-1} \delta_k
  $$

* This identity is not exact if V is updated during the episode (as it is in TD(0) ), but if the step sixe is small then it may still hold approximately.

**Q: -**If V changes during the episode, then (6.6) only holds approximately; what
would the di↵erence be between the two sides? Let V t denote the array of state values
used at time t in the TD error (6.5) and in the TD update (6.2). Redo the derivation
above to determine the additional amount that must be added to the sum of TD errors
in order to equal the Monte Carlo error.

![Exercise 6.1](https://github.com/Xingtao/ReinforceLearningIntro/raw/master/solutions/chapter6/figures/exercise_6-1.png)

## Sarsa Learning : On-policy TD control

* In Sarsa learning we learn an action-value function rather than a state-value function.

  * For an on-policy method we must estimate $q_{\pi}(s,a)$ for the current behaviour policy $\pi$ and for all states s and action a. This can be done using essentially the same TD method described above for learning $v_\pi$ 

  $$
  Q(S_t,A_t) <- Q(S_t,A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1},A_{t+1} - Q(S_t, A_t))]
  $$

  The name SARSA has been derived from the quintuple  $(S_t , A_t , R_{t+1} , S_{t+1} , A_{t+1} )$

* In Monte Carlo method sometimes termination won't be guaranteed for all policies. If a policy was ever found that caused the agent to stay in the same state, then the next episode would never end. Online learning methods such as Sarsa do not have this problem because they quickly learn during methods such as Sarsa do not have this problem because they quickly learn during the episode than such policies are poor, and switch to something else.

## Q learning : Off-policy TD control

* In this case, the learned action-value function, Q, directly approximates $q_{*}$, the optimal action-value function, independent of the policy being followed.

* The policy still has an effect in that it determines which state-action pairs are visited and updated

* Q-learning is called off-policy because the updated policy is different from the behavior policy, so Q-Learning is off-policy. In other words, it estimates the reward for future actions and appends a value to the new state without actually following any greedy policy.

  [Ex 6.12](https://ai.stackexchange.com/questions/21044/are-q-learning-and-sarsa-the-same-when-action-selection-is-greedy/21070)

## [Gym Tutorial](https://medium.com/@ashish_fagna/understanding-openai-gym-25c79c06eccb)

## [Deep RL ]([https://medium.com/deep-math-machine-learning-ai/ch-13-deep-reinforcement-learning-deep-q-learning-and-policy-gradients-towards-agi-a2a0b611617e)

