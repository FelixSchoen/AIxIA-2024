import enum


class AttentionOutputMode(enum.Enum):
    OUTPUT_ONLY = "output_only",
    SCORES_ONLY = "scores_only",  # Pre Softmax
    WEIGHTS_ONLY = "weights_only",
    OUTPUT_AND_WEIGHTS = "output_and_weights"
