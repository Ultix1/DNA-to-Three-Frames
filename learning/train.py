
import os
import numpy as np
import time
from network import DDDQN
from main_agent import Agent
from environment import Environment

with open('data/DNA.txt', 'r') as file:
    dna_sequence = file.read().strip()

with open('data/AA.txt', 'r') as file:
    protein_sequence = file.read().strip()

params = {
    'epsilon' : 0.99999,
    'epsilon_min': 0.01,
    'decay' : 0.99,
    'gamma' : 0.99,
    'buffer_size' : 50000,
    'max_ep' : 500,
    'batch_size' : 64,
    'train_freq' : 100,
    'tau': 0.01
}

environment = Environment(dna_sequence, protein_sequence)
checkpoint_paths = ["./saved_weights/main/main_checkpoint.h5", "./saved_weights/target/target_checkpoint.h5"]

actions = [0, 1, 2, 3]
learning_rate = 0.001
input_shape = (4, 21, 1)

MainQN = DDDQN(learning_rate, len(actions), input_shape)
TargetQN = DDDQN(learning_rate, len(actions), input_shape)
agent = Agent(MainQN, TargetQN, environment, params, actions)

def save_params(episode):
    file = open("./saved_weights/params.txt", "w")
    file.write(f"Episode: {episode}, Epsilon: {agent.epsilon}")
    file.close()

def record_results(episode, score, reward, duration, steps):
    file = open("./saved_weights/training_results.txt", "a")
    file.write(f"Episode: {episode}\n\tScore: {score}, Reward: {reward}, Time Taken: {duration}, Steps Taken: {steps} \n")
    if(episode == params['max_ep']):
        file.write("====================\n====================")
    file.close()






# Training Loop
episodes = 1

# Check if there are saved weights, load weights into networks
resume = (os.path.isfile(checkpoint_paths[0]) and os.path.isfile(checkpoint_paths[1]))
prev_eps = 0
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

# Let Agent Explore (Do random )
else:
    print("\n\nStarting Explore Step...\n")
    agent.explore(reps=100)


# Testing Alignment
agent.test()
exit()


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

    # Save Weights every 10 Episodes
    if(episodes > 0 and episodes % 10 == 0):
        agent.mainQN.model.save_weights(checkpoint_paths[0])
        agent.targetQN.model.save_weights(checkpoint_paths[1])
        save_params(episodes)

    # Record results of agent
    record_results(episodes, score, reward, duration, steps)

    episodes += 1

