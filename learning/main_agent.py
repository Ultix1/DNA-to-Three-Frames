
import tensorflow as tf
import numpy as np
from environment import Environment
from experience_buffer import Experience_Buffer
from keras import Model
from network import DDDQN

class Agent():

    def __init__(self, mainQN : DDDQN, targetQN : DDDQN, env : Environment, params : dict, actions: list):

        self.mainQN = mainQN
        self.targetQN = targetQN
        self.env = env
        self.actions = actions

        self.dna_seq = env.dna_sequence
        self.pro_seq = env.protein_sequence

        self.params = params
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.bufferSize = params['buffer_size']
        self.epsilon_decay = params['decay']
        self.pre_train = params['pre_train']
        self.batchSize = params['batch_size']
        self.train_freq = params['train_freq']

        self.episodeBuffer = Experience_Buffer(self.bufferSize)
        self.total_steps = 0

    def reset(self):
        # self.episodeBuffer = Experience_Buffer(self.bufferSize)
        self.epsilon = self.params['epsilon']
        self.env.reset()
        self.total_steps = 0

    def play(self):

        total_reward = 0
        total_score = 0
        done = False

        while not done:
            # Get current state
            state = self.env.get_state()

            # Get predicted action
            action = self.get_action(state)
            
            # Get results of action
            score, reward, done, next_state = self.env.step(action)

            # Decay Epsilon After Pretraining Steps
            if self.total_steps >= self.pre_train:
                if(self.total_steps % self.train_freq == 0):
                    self.train()

            # Append to Buffer
            self.episodeBuffer.add(state, action, reward, next_state, done)

            total_reward += reward
            total_score += score

            self.total_steps += 1

            # if self.total_steps % 50 == 0:
            #     print(f"Step: {self.total_steps}, Total Score={total_score}, Total Reward={total_reward}, Action={action}\n")

        return total_score, total_reward, self.total_steps


    def train(self):
        states, actions, rewards, next_states, dones = self.episodeBuffer.sample(self.batchSize)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        for i in np.arange(self.batchSize):
            next_q_val_main = self.mainQN.model.predict(next_states[i], verbose=0)

            next_q_val_target = self.targetQN.model.predict(next_states[i], verbose=0)

            action = np.argmax(next_q_val_main)

            target_q = rewards[i] + self.gamma * next_q_val_target[0][action] * (1-dones[i])
            
            next_q_val_main[0][action] = target_q
            
            self.mainQN.model.train_on_batch(states[i], next_q_val_main)


    def get_action(self, state : np.ndarray = None):
        if (np.random.rand() < self.epsilon) or (self.total_steps < self.pre_train):
            self.epsilon *= self.epsilon_decay
            return np.random.choice(self.actions)
        
        else:
            q_vals = self.mainQN.model.predict(state, verbose=0)
            return np.argmax(q_vals)