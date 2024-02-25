
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
import numpy as np


class DDDQN():

    def __init__(self, learning_rate: float, action_size: int, input_shape: tuple, dueling: bool) -> None:
        self.input_shape = input_shape
        self.act_size = action_size
        self.optimizer = Adam(learning_rate)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv2D(16, kernel_size=(3,  3), activation='relu', padding='same', input_shape=self.input_shape),
            Conv2D(32, kernel_size=(3,  3), activation='relu', padding='same'),
            Conv2D(64, kernel_size=(3,  3), activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.act_size, activation='linear')
        ])

        model.compile(optimizer=self.optimizer, loss='mse')
        # model.summary()
        
        return model
    
    # def update_target_network(self):
    #     self.target_network.set_weights(self.main_network.get_weights())

    # def Main_Predict(self, state):
    #     return self.main_network.predict(state, verbose=0)
    
    # def Target_Predict(self, state):
    #     return self.target_network.predict(state, verbose=0)


