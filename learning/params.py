from utils.constants import Action

PARAMS = {
    'epsilon' : 0.99999,            # Starting Epsilon
    'epsilon_min': 0.01,            # Minimum Epsilon
    'decay' : 0.99,                 # Epsilon Decay
    'gamma' : 0.99,                 # Discount Factor for target q vals
    'buffer_size' : 50000,          # Size of Buffer
    'max_ep' : 1000,                # Max training episodes
    'batch_size' : 64,              # Batch size for training
    'train_freq' : 100,             # How Many Steps before Updating Main Q-Network
    'tau': 0.01,                    # Discount Factor for Updating Target Q-Network
    'input_shape' : (8, 22, 1),     # Input Shape
    'lr' : 0.001,
    'actions' : [
        Action.MATCH.value, 
        Action.FRAMESHIFT_1.value, 
        Action.FRAMESHIFT_3.value, 
        Action.INDEL.value, 
        Action.MISMATCH.value,
    ],
}

