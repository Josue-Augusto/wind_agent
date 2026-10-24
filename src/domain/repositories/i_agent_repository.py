from abc import ABC, abstractmethod

from src.domain.entities.agent_state import AgentState


class IAgentRepository(ABC):
    @abstractmethod
    async def start_session(self) -> None:
        """Start a voice session."""

    @abstractmethod
    async def stop_session(self) -> None:
        """Stop the active voice session."""

    @abstractmethod
    def get_state(self) -> AgentState:
        """Return the current state of the voice agent."""
