## PACMAN-using-DRL
This is a project mentored by `Gagan Jain` and `Utkarsh Agarwal` for WNCC's Seasons of Code 2021

### Mentees
* Utkarsh Ranjan - 200050147
<!-- <insert-your-details> -->

### Motivation
Reinforcement Learning (RL) is a field of Artificial Intelligence where an agent learns by interacting with its environment and receiving a reward/penalty for its actions. RL has recently started receiving a lot more attention, owing to the famous victory by an RL agent over the world champion in the game of “Go”. This repo containsthe project aimed to implement RL algorithm on OpenAI's [`PACMAN`](https://gym.openai.com/envs/MsPacman-v0/) and get us familiar with the field.

### How we did this project
We first read the 3 chapters of the book `Sutton and Barto` and learnt python. This was done before the endsem. After endsem we started the implementation, first we wrote a code for gridworld (using [this](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html/) site as our metric) 
Once that was done we installed the [`gym`](https://gym.openai.com/) environment. Read about the  Deep RL architecture and implemented it on cartpole-v0. After that we added a replay buffer and dual network. 
Once this was done we moved towards advanced algorithms read [these](https://github.com/TheSmilingSky/PACMAN-using-DRL/tree/main/research-papers) papers found some implementations online, did some hyperparameter search and started training the agent on colaboratory.

### What we learned
* Implementation of standard python libraries like numpy, matplotlib, 
* Fundamentals of RL algorithms, MDP,TD, Monte Carlo, Q learning, Sarsa Learning
* Usage of ML frameworks for deepLearning like tensorflow, keras
* Familiarising with open ai’s gym a toolkit for developing and comparing RL algorithms
* Advanced and recent developments in RL like NoisyNets , RainbowNets
* Fundamentals involved while training an RL agent like tuning hyperparameter, loading and saving a deep learning model

### Contains
* A folder of k-bandit assignment (assignment)
* A folder of grid world assignment
* A folder of gym-code (atari-games)
* A folder of research-papers 
* [Code](https://github.com/TheSmilingSky/PACMAN-using-DRL/blob/main/pacman_NoisyNet_n_step_PDD_DQN.ipynb) with implementation of research paper
* [Code](https://github.com/TheSmilingSky/PACMAN-using-DRL/blob/main/pacman_NoisyNet_n_step_PDD_DQN.ipynb) used for training (an intermediate model to final code)
* Other codes are intermediate networks written during development

### How to use
* [This](https://github.com/TheSmilingSky/PACMAN-using-DRL/blob/main/pacman_NoisyNet_n_step_PDD_DQN.ipynb) is the final code written by [Utkarsh Ranjan](https://github.com/geekyuttu). It needs to be run on colaboratory for its successful execution.      
* It is necessary to mount gdrive and make folders buffers,models,cum_rewards and plots at path `'/content/drive/MyDrive/pacman_SOC_outputs/'`
* In[1] installs all the dependencies required to run the gym environment in colab which includes ROM for atari-game and Ipython display dependencies.
* Further this code is well commented the network used is NoisyNet_Dueling though there are other networks made while development
* In[2], In[3] are for Ipython display
* All hyperparameters are in In[94]

<!-- * [this](https://github.com/TheSmilingSky/PACMAN-using-DRL/blob/main/cart-pole.ipynb) code is written by [<insert name>](https://github.com/TheSmilingSky/PACMAN-using-DRL/blob/main/cart-pole.ipynb)
  <insert-description>

* [this](https://github.com/TheSmilingSky/PACMAN-using-DRL/blob/main/pacman_dqn.ipynb) code is written by [<insert name>](<insert handle>)
    <insert-description> -->
      
### Results
Mp4 initially [Ep 1](https://drive.google.com/file/d/1jtVG4gNwlWYwmyd5j6CoE_dnyFER7z3a/view?usp=sharing/)

Mp4 after training for [Ep 270](https://drive.google.com/file/d/1Ixl9qIoHsLYO3sbWxY4Jnf3jFbG-NcU6/view?usp=sharing/)

It can be properly observed how the agent learned to avoid the ghost after she died in her first attempt. Further the agent learned to eat the fruit to repel the ghosts and earn points by eating them.
      
[This](https://github.com/TheSmilingSky/PACMAN-using-DRL/tree/main/output-pacman/e-decay_DDQN) contains all the outputs- models, plots and videos.Plots are not continuous due to non-continuous training on colab.



