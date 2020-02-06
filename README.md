# Deep Q-learning for maze solving

A simple implementation of DQN that uses PyTorch and a fully connected neural network to estimate the q-values of each state-action pair.

The environment is a maze that is randomly generated using a deep-first search algorithm to estimate the Q-values. Four moves are possible for the agent (up, down, left and right), whose objective is to reach a predetermined cell. The agent implements either an epsilon-greedy policy or a softmax behaviour policy with temperature equal to epsilon. After each episode, the starting position is sampled in such a way that at the beginning of the training the agent explores the area surrounding the goal, and as the training goes on it will explore further and further areas of the maze.

A convolutional neural network is also implemented for completeness.
