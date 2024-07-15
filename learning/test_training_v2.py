import os
from models_v2.environment import Environment
from models_v2.main_agent import Agent
from models_v2.network import DDDQN
from params import PARAMS

dna_dir = "data/dna"
protein_dir = "data/proteins"
save_dir = "results/debug"

dna_list = []
protein_list = []

for fn in os.listdir(dna_dir):
    dna_list.append(f"{dna_dir}/{fn}")

for fn in os.listdir(protein_dir):
    protein_list.append(f"{protein_dir}/{fn}")

input_shape = PARAMS['input_shape']
actions = PARAMS['actions']
learning_rate = PARAMS['lr']

MainQN = DDDQN(learning_rate, len(actions), input_shape)
TargetQN = DDDQN(learning_rate, len(actions), input_shape)

environment = Environment(window_size=PARAMS['window_size'])
agent = Agent(MainQN, TargetQN, environment, PARAMS, actions)

# Load Weights
checkpoint_paths = ["./saved_weights/main/main_checkpoint.weights.h5", "./saved_weights/target/target_checkpoint.weights.h5"]

if(os.path.isfile(checkpoint_paths[0]) and os.path.isfile(checkpoint_paths[1])):
    
    # Set defailt epsilon value and load weights
    agent.epsilon = 0.01
    agent.load_weights(checkpoint_paths[0], checkpoint_paths[1])

    for i in range(len(dna_list)):
        file_1 = open(dna_list[i], 'r')
        file_2 = open(protein_list[i], 'r')

        dna = file_1.read().strip()
        protein = file_2.read().strip()

        file_1.close()
        file_2.close()

        environment.set_seq(dna, protein)

        print(f"For DNA and Protein Sequence {i + 1}:")
        agent.test(save_dir, dna_list[i], protein_list[i], save=True)
        print("")
        
        if(i == 14):
            break