# **Multi Agents Constrained Attention Actor Critic**

Code for our paper titled "Attention Actor-Critic algorithm for Multi-Agent Constrained Co-operative Reinforcement Learning". 

Base code is used from Shariq Iqbal's PyTorch implementation of MAAC:- https://github.com/shariqiqbal2810/MAAC

### Requirements 
#
Python 3.7.1 

PyTorch, version: 1.1.0 

[OpenAI Gym](https://github.com/openai/gym), version: 0.9.4

[Tensorboard](https://github.com/tensorflow/tensorboard), version: 2.1.0

[Tensorboard-PyTorch](https://github.com/lanpa/tensorboardX), version: 1.9

### Training 
#
To train on default hyperparameters(used for the pre-trained models), run the following commands:

For Constrained Cooperative Navigation: `python main.py`

For Constrained Cooperative Treasure Collection: `python main_treasure_collection.py`

