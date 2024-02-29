
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Lambda, Input as K_Input
from keras.optimizers import Adam
from tensorflow import reduce_mean
import numpy as np


class DDDQN():

    def __init__(self, learning_rate: float, action_size: int, input_shape: tuple) -> None:
        """
        Initialize Dueling Double Deep Q-Network

        Args:
            learning_rate (float): Learning Rate to be used by optimizer
            action_size (int): Number of possible actions
            input_shape (tuple): Input shape of the first input
        """
        self.optimizer = Adam(learning_rate)
        self.act_size = action_size
        self.model = self.build_model(input_shape, action_size)

    def build_model(self, input_shape, n_actions):

        inputs = K_Input(shape=input_shape)

        # Convolutional Layer
        conv_1 = Conv2D(32, kernel_size=(3,  3), activation='relu', padding='same')(inputs)
        conv_2 = Conv2D(64, kernel_size=(3,  3), activation='relu', padding='same')(conv_1),
        conv_3 = Conv2D(128, kernel_size=(3,  3), activation='relu', padding='same')(conv_2[0]),
        flatten = Flatten()(conv_3[0])

        # Get State Value
        state_value = Dense(256, activation='relu')(flatten)
        state_value = Dense(128, activation='relu')(state_value)
        state_value = Dense(1, activation='linear')(state_value)

        # Get Advantage Value
        advantage = Dense(256, activation='relu')(flatten)
        advantage = Dense(128, activation='relu')(advantage)
        advantage = Dense(self.act_size, activation='linear')(advantage)

        # Combine Both
        advantage_mean = reduce_mean(advantage, axis=-1, keepdims=True)
        q_vals = state_value + (advantage - advantage_mean)

        return Model(inputs=inputs, outputs=q_vals)

    # model = Sequential([
    #     Conv2D(32, kernel_size=(3,  3), activation='relu', padding='same', input_shape=self.input_shape),
    #     Conv2D(64, kernel_size=(3,  3), activation='relu', padding='same'),
    #     Conv2D(128, kernel_size=(3,  3), activation='relu', padding='same'),
    #     Flatten()

    #     Dense(128, activation='relu'),
    #     Dense(64, activation='relu'),
    #     Dense(self.act_size, activation='linear')
    # ])
    # model.compile(optimizer=self.optimizer, loss='mse')

    # def dueling_predict(self, state):

    #     prediction = self.model.predict(state, verbose=0)

    #     # State Value
    #     state_value = Dense(256, activation='relu')(prediction)
    #     state_value = Dense(128, activation='relu')(state_value)
    #     state_value = Dense(1, activation='linear')(state_value)

    #     # Advantage
    #     advantage = Dense(256, activation='relu')(prediction)
    #     advantage = Dense(128, activation='relu')(advantage)
    #     advantage = Dense(self.act_size, activation='linear')(advantage)
    #     advantage_mean = reduce_mean(advantage, axis=-1, keepdims=True)

    #     output = state_value + (advantage - advantage_mean)

    #     return output.numpy()
