## PACMAN-using-DRL
This is a project mentored by `Gagan Jain` and `Utkarsh Agarwal` for WNCC's Seasons of Code 2021

<img src=https://github.com/TheSmilingSky/PACMAN-using-DRL/blob/main/images/pacman.png align=middle></img>

### Mentees
* Utkarsh Ranjan - 200050147
* Alakh Agrawal  - 200040018
* Nikhil Kaniyeri - 200070050
* Akshat Gautam - 190110004
<!-- <insert-your-details> -->

### Motivation
Reinforcement Learning (RL) is a field of Artificial Intelligence where an agent learns by interacting with its environment and receiving a reward/penalty for its actions. RL has recently started receiving a lot more attention, owing to the famous victory by an RL agent over the world champion in the game of “Go”. This repo containsthe project aimed to implement RL algorithm on OpenAI's [`PACMAN`](https://gym.openai.com/envs/MsPacman-v0/) and get us familiar with the field.

### How we did this project
We first read the 3 chapters of the book `Sutton and Barto` and learnt python. This was done before the endsem. After endsem we started the implementation, first we wrote a code for gridworld (using [this](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html/) site as our metric) 
Once that was done we installed the [`gym`](https://gym.openai.com/) environment. Read about the  Deep RL architecture and implemented it on cartpole-v0. After that we added a replay buffer and dual network. 
Once this was done we moved towards advanced algorithms read [these](https://github.com/TheSmilingSky/PACMAN-using-DRL/tree/main/research-papers) papers found some implementations online, did some hyperparameter search and started training the agent on colaboratory.

### What we learned
* You can make impossible things possible if you learn from tons of your mistake and correct them on your way. #Reinforcement Learning
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
  <insert-description>-->


* [This](https://github.com/TheSmilingSky/PACMAN-using-DRL/blob/main/pacman_dqn.ipynb) code is written by [Nikhil Kaniyeri.](https://github.com/kaniyeri)
* It runs directly on Visual Studio Code. Requires installation of gym and gym[atari] locally. 
* You also need to download the ROM for Pacman and import it into the library. This is a one time process and is detailed [here.](https://github.com/openai/atari-py/#roms)
      
### Results
Mp4 initially [Ep 1](https://drive.google.com/file/d/1jtVG4gNwlWYwmyd5j6CoE_dnyFER7z3a/view?usp=sharing/)

Mp4 after training for [Ep 270](https://drive.google.com/file/d/1Ixl9qIoHsLYO3sbWxY4Jnf3jFbG-NcU6/view?usp=sharing/)
It can be properly observed how the agent learned to avoid the ghost after she died in her first attempt. Further the agent learned to eat the fruit to repel the ghosts and earn points by eating them.

Mp4 after training for [Ep 600](https://drive.google.com/file/d/19w4nzh_bPGFAx7cbQVjtJ4YaKhGJQmMA/view?usp=sharing)
By now agent have started avoiding ghost (clear in this video) ,at the same time she has learned not to avoid them when they are blue.

[This](https://github.com/TheSmilingSky/PACMAN-using-DRL/tree/main/output-pacman/e-decay_DDQN) contains all the outputs- models, plots and videos.Plots are not continuous due to non-continuous training on colab.

### Appendix
* Gridworld reference - https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
* OpenAI Gym in Colab - https://colab.research.google.com/github/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_12_01_ai_gym.ipynb#scrollTo=dnID4yguIeX7
* Hyperparameters - https://paperswithcode.com/paper/rainbow-combining-improvements-in-deep/review/?hl=19878
* https://medium.com/deep-math-machine-learning-ai/ch-13-deep-reinforcement-learning-deep-q-learning-and-policy-gradients-towards-agi-a2a0b611617e
* Preprocessing Image - https://www.datahubbs.com/deepmind-dqn/
* https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
* https://acsweb.ucsd.edu/~wfedus/pdf/replay.pdf
* https://towardsdatascience.com/4-ways-to-boost-experience-replay-999d9f17f7b6
* https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
* https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
* https://paperswithcode.com/method/prioritized-experience-replay#
* https://www.youtube.com/watch?v=MqZmwQoOXw4
* https://github.com/the-computer-scientist/OpenAIGym/blob/master/PrioritizedExperienceReplayInOpenAIGym.ipynb
* https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/
* https://towardsdatascience.com/introduction-to-reinforcement-learning-rl-part-7-n-step-bootstrapping-6c3006a13265
* https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning/blob/master/05_multistep_td.py
* https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning/blob/d1c191e7bfbb44357a4066ced3b96fa8c847875a/07_noisynet.py#L309
