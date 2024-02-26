
import numpy as np
from network import DDDQN
from main_agent import Agent
from environment import Environment

with open('data/DNA.txt', 'r') as file:
    dna_sequence = file.read().strip()

with open('data/AA.txt', 'r') as file:
    protein_sequence = file.read().strip()

params = {
    'epsilon' : 0.99,
    'decay' : 0.05,
    'gamma' : 0.99,
    'buffer_size' : 50000,
    'pre_train' : 50,
    'max_ep' : 100,
    'batch_size' : 32,
    'train_freq' : 100
}

environment = Environment(dna_sequence, protein_sequence, 0)

actions = [0, 1, 2, 3]
learning_rate = 0.001
input_shape = (12, 8, 1)

MainQN = DDDQN(learning_rate, len(actions), input_shape, False)
TargetQN = DDDQN(learning_rate, len(actions), input_shape, False)
agent = Agent(MainQN, TargetQN, environment, params, actions)

# TODO Train Loop
episodes = 1
while episodes < params['max_ep']:

    score, reward, steps = agent.play()
    print(f"Episode: {episodes}\n\tScore: {score}, Reward: {reward}, Number of Steps: {steps}\n")

    agent.reset()

    episodes += 1