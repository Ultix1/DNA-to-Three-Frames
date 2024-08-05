import os
from models_v2.environment import Environment
from models_v2.main_agent import Agent
from models_v2.network import DDDQN
from utils.aligner import ThreeFrameAligner
from utils.fasta_reader import read_fasta
from params import PARAMS

dna_dir = "data/dna"
protein_dir = "data/proteins"
save_dir = "results/accuracy_tests"

# Parameters
input_shape = PARAMS['input_shape']
actions = PARAMS['actions']
learning_rate = PARAMS['lr']

# Load Weights
checkpoint_paths = ["./saved_weights/main/main_checkpoint.weights.h5", "./saved_weights/target/target_checkpoint.weights.h5"]

if(os.path.isfile(checkpoint_paths[0]) and os.path.isfile(checkpoint_paths[1])):
    MainQN = DDDQN(learning_rate, len(actions), input_shape)
    TargetQN = DDDQN(learning_rate, len(actions), input_shape)

    environment = Environment(window_size=PARAMS['window_size'])
    agent = Agent(MainQN, TargetQN, environment, PARAMS, actions)
    aligner = ThreeFrameAligner()

    # Set defailt epsilon value and load weights
    agent.epsilon = 0.01
    agent.load_weights(checkpoint_paths[0], checkpoint_paths[1])

    # Set Target DNA
    with open("fasta_tests/test_1/translated_dna.txt", 'r') as target_dna_file:
        target_dna = target_dna_file.read().strip()

    # Load 1000 Reference Proteins
    fasta_file = "fasta_tests/fasta_files/fruit_fly.fasta"
    reference_proteins = read_fasta(fasta_file, max_size=1000, protein_len_range=(100, 150))

    i = 0
    for key, protein in reference_proteins.items():
        environment.set_seq(target_dna, protein)

        score, reward = agent.test("", "", "", save=False)
        aligner_score, _, __ = aligner.align(target_dna, protein, debug=False)
        percent_diff = (abs(aligner_score - score) / ((score + aligner_score)/2)) * 100

        with open(f"{save_dir}/results.txt", 'a') as file:
            output = "Sequence {:<3}: {:<10} {:<10} {:<10}\n".format(i + 1, score, aligner_score, percent_diff)
            file.write(output)
        
        i += 1
    
    # for i in range(len(reference_proteins)):
        
    #     file_2 = open(protein_list[i], 'r')

    #     dna = file_1.read().strip()
    #     protein = file_2.read().strip()

    #     file_1.close()
    #     file_2.close()

    #     environment.set_seq(dna, protein)

    #     print(f"For DNA and Protein Sequence {i + 1}:")
    #     score, reward = agent.test(save_dir, dna_list[i], protein_list[i], save=False)
    #     file = open(f"{save_dir}/results.txt", 'a')

    #     score_2, _, __ = aligner.align(dna, protein, debug=False)
    #     percent_diff = (abs(score_2 - score) / ((score + score_2)/2)) * 100
        
    #     output = "Sequence {:<3}: {:<10} {:<10} {:<10}\n".format(i + 1, score, score_2, percent_diff)
    #     file.write(output)
    #     file.close()
        
    #     print("\n")