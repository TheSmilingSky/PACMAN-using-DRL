**April 2nd**
* Python - [Sentdex](https://www.youtube.com/watch?v=eXBD2bB9-RA&list=PLQVvvaa0QuDeAams7fkdcwOGBpGdHpXln)
* Python Practice - [WNCC's Code in Quarantine](https://github.com/wncc/CodeInQuarantine/tree/master/Week_1_Python)
* Introduction to RL - [Sutton and Barto Chapter 01](http://incompleteideas.net/book/RLbook2018trimmed.pdf)

**April 7th**
* Bandits and an introductory coding problem - [Sutton and Barto Chapter 02, 03](http://incompleteideas.net/book/RLbook2018trimmed.pdf). Try to implement a simple bandit instance with 10 arms and randomly generated but fixed true means of the arms between 0 and 1. Use epsilon-greedy method and record the regret for these values of epsilon: 0.001, 0.01, 0.1, 0.9.

**And we're back again after the break**  
**May 16th**
* Markov Decision Processes - [Sutton and Barto Chapter 03](http://incompleteideas.net/book/RLbook2018trimmed.pdf)
* TD Learning and Q-learning - [Sutton and Barto Chapter 06, Sec 6.1 to 6.5](http://incompleteideas.net/book/RLbook2018trimmed.pdf)
* Intro to deep Q-learning and experience replay (hope this'll give more clarity about the project flow) - [Blog on Intro to Deep Q-learning](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/)
* PacMan environment - [Gym PacMan](https://gym.openai.com/envs/MsPacman-v0/), [Gym Docs for visualization](https://gym.openai.com/docs/)
* Video lectures (optional) - [Deepmind 2015 by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

**May 31st**
* Start coding for the network! More details in the meet :)

**July 13th**

Alright guys! Time to complete things. Here's the promised explanation with some additional challenges for you!
The key idea is just to extend the idea of learning in the gridworld environment to the pacman environment. The first step is to construct the state space. Now how does one do that? We can access the environment renders from the gym environment and use that directly as the state space representation. But now, in order to work with that, instead of tabulating the Q-values, we will have to approximate because of the large state space dimensionality. For this, we use a convolutional neural network that tried to estimate the Q values of the states, which are the different locations the agent can take in the game. A network like this is called a DQN (Deep Q-network) network. The notion of rewards is something that I think you can define by yourselves. This is all that one needs to do to have a basic learning agent. But in order to improve upon the results, there's a lot more that can be tried. 

There are other variants that you might wanna try which include DDQN (Double DQN). The idea here is to keep the learned Q-values under check. Why? The reason is because a single DQN is known to overestimate the Q-values when it learns. This can be countered by using a second network that learns a policy and then the update rules use the Q-values derived from the policies. Another issue that a DQN might face is the lack of experience. Why? Because the state space is so dense that it does not encounter a lot of them even after a lot of episodes. Hence, the need for a lot of training. Instead, we can use experience replay which in some sense simulates the agent steps using the current state and learns the policy from the rewards of those steps, without actually doing that in the game. Sounds confusing, right? You can get a very clear idea of what experience replay does by reading a section dedicated to it in Sutton and Barto. 
