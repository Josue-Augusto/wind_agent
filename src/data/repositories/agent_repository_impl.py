import asyncio
from collections.abc import Callable
from threading import Lock

from src.data.datasources.local_pipecat_datasource import LocalPipecatDataSource
from src.domain.entities.agent_state import AgentState
from src.domain.repositories.i_agent_repository import IAgentRepository


class AgentRepositoryImpl(IAgentRepository):
    def __init__(
        self,
        datasource: LocalPipecatDataSource,
        on_state_change: Callable[[AgentState], None] | None = None,
    ) -> None:
        self._datasource = datasource
        self._on_state_change = on_state_change
        self._state = AgentState.IDLE
        self._state_lock = Lock()
        self._session_lock = asyncio.Lock()
        self._datasource.set_state_listener(self._set_state)

    async def start_session(self) -> None:
        async with self._session_lock:
            if self.get_state() is not AgentState.IDLE:
                raise RuntimeError("The voice agent is already active.")
            self._set_state(AgentState.PROCESSING)
            try:
                await self._datasource.start_session()
            except (Exception, asyncio.CancelledError):
                self._set_state(AgentState.IDLE)
                raise

    async def stop_session(self) -> None:
        async with self._session_lock:
            if self.get_state() is AgentState.IDLE:
                return
            await self._datasource.stop_session()
            self._set_state(AgentState.IDLE)

    def get_state(self) -> AgentState:
        with self._state_lock:
            return self._state

    def _set_state(self, state: AgentState) -> None:
        with self._state_lock:
            self._state = state
        if self._on_state_change is not None:
            self._on_state_change(state)
