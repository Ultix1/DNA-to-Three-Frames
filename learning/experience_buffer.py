import random
from collections import deque
import numpy as np

class Experience_Buffer:
    def __init__(self, capacity, batch_size=50000):
        """
        Initialize the replay buffer with a given capacity.

        Parameters:
        - capacity (int): The maximum number of transitions to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)
        self.batchSize = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.

        Parameters:
        - state (np.array): The starting state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (np.array): The next state.
        - done (bool): Whether the episode has finished.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Parameters:
        - batch_size (int): The size of the batch to sample.

        Returns:
        - A list of tuples containing states, actions, rewards, next_states, and dones.
        """
        mini_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*mini_batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns:
        - The number of transitions in the buffer.
        """
        return len(self.buffer)
