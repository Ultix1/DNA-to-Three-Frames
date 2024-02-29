
import tensorflow as tf
import numpy as np
from environment import Environment
from experience_buffer import Experience_Buffer
from keras import Model
from keras.optimizers import Adam
from network import DDDQN

class Agent():

    def __init__(self, mainQN : DDDQN, targetQN : DDDQN, env : Environment, params : dict, actions: list):
        """
        Initializes the agent that will interact with the environment

        Args:
            mainQN (DDDQN): Dueling Double Deep Q-Network
            targetQN (DDDQN): Dueling Double Deep Q-Network
            env (Environment): Chosen environment for agent
            params (dict): User defined parameters
            actions (list): List of possible actions
        """
        self.mainQN = mainQN
        self.targetQN = targetQN
        self.env = env
        self.actions = actions
        self.optimizer = Adam(0.001)

        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.gamma = params['gamma']
        self.bufferSize = params['buffer_size']
        self.epsilon_decay = params['decay']
        self.batchSize = params['batch_size']
        self.train_freq = params['train_freq']

        self.episodeBuffer = Experience_Buffer(self.bufferSize)
        self.episode = 0
        self.total_steps = 0

    def reset(self):
        """
        Resets the environment and total number of steps
        """
        self.env.reset()
        self.episode += 1
        self.total_steps = 0

    def play(self):
        """
        Agent interacts with the environment until the environment reaches its finished state

        Returns:
            Total Score: Total reward received by the agent after finishing the task
            Total Reward: Total reward received by the agent after finishing the task
            Total Steps: Number of steps taken by the agent before finishing
        """
        total_reward = 0
        total_score = 0
        done = False

        while not done:
            # Get current state
            state = self.env.get_state()

            # Get predicted action
            action = self.get_action(tf.expand_dims(state, axis=0))

            # Get results of action
            score, reward, done, next_state = self.env.step(action) if (self.total_steps > 0) else self.env.first_step(action)

            # Train main_network every select steps
            if(self.total_steps > 0 and self.total_steps % self.train_freq == 0):
                self.train()

            # Append to Buffer
            self.episodeBuffer.add(state, action, reward, next_state, done)

            total_reward += reward
            total_score += score

            self.total_steps += 1

        return total_score, total_reward, self.total_steps


    def train(self):
        """
        Updates the weights of the main network using recorded previous steps
        """
        states, actions, rewards, next_states, dones = self.episodeBuffer.sample(self.batchSize)

        # Target_QN Predictions on next_state
        next_q_val_target = self.targetQN.model(next_states)
        
        # Get Best Action
        best_actions = tf.argmax(next_q_val_target, axis=1, output_type=tf.int32)

        # Get Target Q-Value
        action_results = tf.gather(next_q_val_target, best_actions, axis=1, batch_dims=1)
        target_q_values = rewards + (self.gamma * action_results * (1-dones))

        indices = tf.range(self.batchSize, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis = 1)

        # Apply new gradients
        self.update_mainQN(states, target_q_values, action_indices)

    @tf.function
    def update_mainQN(self, input, target_q, action_indices):
        with tf.GradientTape() as tape:
            prediction = tf.gather_nd(self.mainQN.model(input), indices=action_indices)
            loss = self.loss_fn(target_q, prediction)
    
        # Calculate New Gradients from loss
        gradients = tape.gradient(loss, self.mainQN.model.trainable_weights)

        # Apply new gradients to model's weights
        self.optimizer.apply_gradients(zip(gradients, self.mainQN.model.trainable_weights))

        return loss

    def soft_update_model(self, tau=0.01):
        """
        Updates the target network using the weights of the main network

        Args:
            tau (float, optional): Discounting factor for updating the weights. Defaults to 0.01.
        """
        for target_weight, local_weight in zip(self.targetQN.model.weights, self.mainQN.model.weights):
            target_weight.assign(tau * local_weight + (1 - tau) * target_weight)

    def get_action(self, state : np.ndarray = None):
        """
        Get action given a state

        Args:
            state (np.ndarray, optional): Input state. Defaults to None.

        Returns:
            Action: Integer value representing the action to be taken
        """
        if (np.random.rand() < self.epsilon and self.epsilon != self.epsilon_min):
            return np.random.choice(self.actions)
        
        else:
            q_vals = self.mainQN.model(state)
            return np.argmax(q_vals)
        
    def load_weights(self, path_1, path_2):
        """
        Load weights into current networks

        Args:
            path_1: Path to main network weights
            path_2: Path to target network weights
        """
        self.mainQN.model.load_weights(path_1)
        self.targetQN.model.load_weights(path_2)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def loss_fn(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
