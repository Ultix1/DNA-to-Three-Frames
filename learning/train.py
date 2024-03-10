
import os
import numpy as np
import time
from utils.constants import Action
from network import DDDQN
from main_agent import Agent
from environment import Environment

def save_params(episode, epsilon):
    file = open("./saved_weights/params.txt", "w")
    file.write(f"Episode: {episode}, Epsilon: {epsilon}")
    file.close()

def record_results(episode, score, reward, duration, steps):
    file = open("./saved_weights/training_results.txt", "a")
    file.write(f"Episode: {episode}\n\tScore: {score}, Reward: {reward}, Time Taken: {duration}, Steps Taken: {steps} \n")
    if(episode == params['max_ep']):
        file.write("====================\n====================\n")
    file.close()


dna_dir = "data/dna"
protein_dir = "data/proteins"

dna_list = []
protein_list = []

for fn in os.listdir(dna_dir):
    with open(f"{dna_dir}/{fn}", 'r') as file:
        dna_list.append(file.read().strip())
        file.close()

for fn in os.listdir(protein_dir):
    with open(f"{protein_dir}/{fn}", 'r') as file:
        protein_list.append(file.read().strip())
        file.close

params = {
    'epsilon' : 0.99999,
    'epsilon_min': 0.01,
    'decay' : 0.99,
    'gamma' : 0.99,
    'buffer_size' : 50000,
    'max_ep' : 5000,
    'batch_size' : 64,
    'train_freq' : 100,
    'tau': 0.01
}

checkpoint_paths = ["./saved_weights/main/main_checkpoint.h5", "./saved_weights/target/target_checkpoint.h5"]

actions = [Action.MATCH.value, Action.FRAMESHIFT_1.value, Action.FRAMESHIFT_3.value, Action.INSERT.value, Action.DELETE.value]
learning_rate = 0.001
input_shape = (8, 21, 1)

MainQN = DDDQN(learning_rate, len(actions), input_shape)
TargetQN = DDDQN(learning_rate, len(actions), input_shape)

environment = Environment()
agent = Agent(MainQN, TargetQN, environment, params, actions)


# <============================================= Training Loop =============================================>

# Check if there are saved weights, load weights into networks
resume = (os.path.isfile(checkpoint_paths[0]) and os.path.isfile(checkpoint_paths[1]))
prev_episodes = 0

# Resume Training
if resume:
    print("\n\nResuming Training...\n")
    file_1 = open("./saved_weights/params.txt", "r")
    file_2 = open("./saved_weights/training_results.txt", "r")
    
    epsilon = float(file_1.readline().split(" ")[-1])
    prev_episodes = int(list(file_2)[-4].split(" ")[-1])

    agent.epsilon = epsilon
    agent.load_weights(checkpoint_paths[0], checkpoint_paths[1])

    file_1.close()
    file_2.close()

# Let Agent Explore (Do random actions)
else:
    print("\n\nStarting Explore Step...\n")
    for i in range(len(dna_list)):
        environment.set_seq(dna_list[i], protein_list[i])
        agent.explore(reps=20)

episodes = 1
i = 0
environment.set_seq(dna_list[i], protein_list[i])

while episodes <= params['max_ep']:
        
    start = time.time()
    score, reward, steps = agent.play()
    duration = time.time() - start
    
    agent.decay_epsilon()
    agent.reset()

    # Print results of agent
    print(f"Episode: {episodes + prev_episodes}\n\tScore: {score}, Reward: {reward}, Time Taken: {duration}, Steps Taken: {steps} \n")
    
    # Update Target Q-Network every 20 Episodes
    if(episodes > 0 and episodes % 20 == 0):
        agent.soft_update_model(params['tau'])
        
        i = (i + 1) % len(dna_list)
        environment.set_seq(dna_list[i], protein_list[i])

    # Save Weights every 10 Episodes
    if(episodes > 0 and episodes % 10 == 0):
        agent.mainQN.model.save_weights(checkpoint_paths[0])
        agent.targetQN.model.save_weights(checkpoint_paths[1])
        save_params(episodes, agent.epsilon)

    # Record results of agent
    record_results(episodes, score, reward, duration, steps)

    episodes += 1

