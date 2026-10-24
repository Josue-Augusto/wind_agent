import asyncio

from src.domain.entities.agent_state import AgentState
from src.domain.repositories.i_agent_repository import IAgentRepository


class ExecuteVoiceSession:
    def __init__(self, agent_repository: IAgentRepository) -> None:
        self._agent_repository = agent_repository
        self._execution_lock = asyncio.Lock()

    async def execute(self) -> AgentState:
        async with self._execution_lock:
            if self._agent_repository.get_state() is AgentState.IDLE:
                await self._agent_repository.start_session()
            else:
                await self._agent_repository.stop_session()
            return self._agent_repository.get_state()
