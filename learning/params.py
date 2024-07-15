from utils.constants import Action

PARAMS = {
    'epsilon' : 0.99999,            # Starting Epsilon
    'epsilon_min': 0.01,            # Minimum Epsilon
    'decay' : 0.995,                # Epsilon Decay
    'gamma' : 0.99,                 # Discount Factor for target q vals
    'buffer_size' : 50000,          # Size of Buffer
    'max_ep' : 2000,                # Max training episodes
    'batch_size' : 64,              # Batch size for training
    'train_freq' : 100,             # How Many Steps before Updating Main Q-Network
    'tau': 0.01,                    # Discount Factor for Updating Target Q-Network
    'window_size' : 2,              # Window Size for Input
    'input_shape' : (12, 23, 1),    # Input Shape is (4 + (w*4), 23, 1) where w is the window size
    'lr' : 0.001,
    'actions' : [
        Action.MATCH.value, 
        Action.FRAMESHIFT_1.value, 
        Action.FRAMESHIFT_3.value, 
        Action.INSERT.value,
        Action.DELETE.value, 
        Action.MISMATCH.value,
    ],
}

