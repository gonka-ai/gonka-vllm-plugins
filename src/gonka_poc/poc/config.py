from enum import Enum


class PoCState(Enum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    STOPPED = "STOPPED"
