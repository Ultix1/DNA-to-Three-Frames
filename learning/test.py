import os
import random
import time
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from models.environment import Environment
from models.main_agent import Agent
from models.network import DDDQN
from params import PARAMS

def getN_chars(string:str, n: int = 10):
    if len(string) >= n:
        start_index = random.randrange(0, len(string) - n)
        selected_chars = string[start_index:start_index + n]
        return start_index, selected_chars

    else:
        return -1, ""

# Load Weights
checkpoint_paths = ["./saved_weights/main/main_checkpoint.h5", "./saved_weights/target/target_checkpoint.h5"]
test_output_path = "results/query_tests"

if __name__ == '__main__':
    if(os.path.isfile(checkpoint_paths[0]) and os.path.isfile(checkpoint_paths[1])):
        fasta_dir = "fasta_tests"
        tests = []
        filenames = []

        # Number of Characters for target
        N_CHARS = 10

        for fn in os.listdir(fasta_dir):
            if(fn != "fasta_files"):
                filenames.append(fn)
                tests.append(f"{fasta_dir}/{fn}")

        input_shape = PARAMS['input_shape']
        actions = PARAMS['actions']
        learning_rate = PARAMS['lr']

        MainQN = DDDQN(learning_rate, len(actions), input_shape)
        TargetQN = DDDQN(learning_rate, len(actions), input_shape)

        environment = Environment()
        agent = Agent(MainQN, TargetQN, environment, PARAMS, actions)

        agent.epsilon = 0.01
        agent.load_weights(checkpoint_paths[0], checkpoint_paths[1])
 
        for index in range(len(tests)):
            file_1 = open(f"{tests[index]}/translated_dna.txt", 'r')
            file_2 = open(f"{tests[index]}/protein.txt", 'r')
            file_3 = open(f"{tests[index]}/id.txt", 'r')

            dna = file_1.read().strip()
            protein = file_2.read().strip()
            id = file_3.read().strip()
            target_index, target_protein = getN_chars(protein, N_CHARS)
            
            file_1.close()
            file_2.close()
            file_3.close()

            # Set DNA
            environment.set_dna(dna)
            results = []

            # Compare to all other proteins
            for i in range(len(tests)):
                protein_file = open(f"{tests[i]}/protein.txt", 'r')
                ref_protein = protein_file.read().strip()
                protein_file.close()

                # Set Protein
                environment.set_protein(ref_protein)
                score, reward = agent.test("", "", save=True)
                
                results.append([reward, score, f"{tests[i]}/protein.txt"])

            save_file = open(f"{test_output_path}/{filenames[index]}.txt", 'a')
            save_file.write(f"Target Protein: {target_protein}, found at Index {target_index}\n")
            save_file.write(f"Origin: {tests[index]}/protein.txt\n")
            save_file.write(f"Protein ID: {id}\n\n")

            for x in results:
                save_file.write(f"Compared with {x[2]}\n")
                save_file.write(f"Alignment Score: {x[1]}\n")
                save_file.write(f"Agent Reward: {x[0]}\n\n")

            save_file.close()

        # for i in range(len(tests)):
        #     file_1 = open(tests[i][0], 'r')
        #     file_2 = open(tests[i][1], 'r')

        #     dna = file_1.read().strip()
        #     protein = file_2.read().strip()

        #     file_1.close()
        #     file_2.close()

        #     index, target_protein = getN_chars(protein, N_CHARS)
        #     environment.set_seq(dna, protein)

        #     print(f"\nTarget Protein: {target_protein}, Index at: {index}")
        #     start = time.time()
        #     agent.find(filename_1=tests[i][0], filename_2=tests[i][1], target_protein=target_protein, protein_len=N_CHARS, save=True)
        #     duration = time.time() - start
        #     print(f"Time Taken: {duration} seconds\n")



# DNA: ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG
# Original Protein: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC => ~0
# Target: CCCCCCCCCC
# 
# Other Protein: AAAAAAAAAAAAAAAAAAAAAAA => -40
# Other Protein: AAAAAAAAAAAAAAAAAAAAAAA => -300
# Other Protein: AAAAAAAAAAAAAAAAAAAAAAA => -50