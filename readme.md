# **Multi Agent Constrained Attention Actor Critic**

Code for our paper [Attention Actor-Critic algorithm for Multi-Agent Constrained Co-operative Reinforcement Learning](https://arxiv.org/abs/2101.02349). 

Base code is adopted from Shariq Iqbal's [PyTorch implementation](https://github.com/shariqiqbal2810/MAAC) of the paper [Actor-Attention-Critic for Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1810.02912.pdf).

### Requirements 
#
Python 3.7.1 

PyTorch, version: 1.1.0 

[OpenAI Baselines](https://github.com/openai/baselines/)

[OpenAI Gym](https://github.com/openai/gym), version: 0.9.4

[Tensorboard](https://github.com/tensorflow/tensorboard), version: 2.1.0

[Tensorboard-PyTorch](https://github.com/lanpa/tensorboardX), version: 1.9

### Training 
#
To train on default hyperparameters, run the following commands:

For Constrained Cooperative Navigation: `python main.py`

For Constrained Cooperative Treasure Collection: `python main_treasure_collection.py`

