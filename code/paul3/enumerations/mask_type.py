from enum import Enum


class MaskType(Enum):
    PADDING = "padding"
    LOOKAHEAD = "lookahead"
    COMBINED = "combined"
    CONSTRAINTS = "constraints"
