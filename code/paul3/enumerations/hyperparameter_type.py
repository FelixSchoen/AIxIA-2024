from enum import Enum


class HPT(Enum):
    # Length of the tensors
    SEQ_LEN = "seq_len"
    # Maximum support sequence length for, e.g., positional encoding
    MAX_LEN = "max_len"
    # Maximum length for, e.g., relative attention
    DISTANCE_LIMIT = "distance_limit"

    # Amount of different tokens (including padding)
    VOCAB_SIZE = "vocab_size"
    SOURCE = "source"
    TARGET = "target"

    # Layers of both encoder and decoder
    NUM_LAYERS = "n_layers"
    # Internal dimension of the model
    D_MODEL = "d_model"
    # Dimension of the feed-forward layer
    D_FF = "d_ff"
    # Number of attention heads
    NUM_HEADS = "n_heads"
    # Dropout applied for training
    DROPOUT_RATE = "dropout_rate"
