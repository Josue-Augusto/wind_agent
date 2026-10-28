from enum import Enum


class AgentState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    LISTENING = "listening"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    SPEAKING = "speaking"
