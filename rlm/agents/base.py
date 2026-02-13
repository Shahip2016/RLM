from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union
from rlm import RLM, RLMConfig, RLMResult

class Agent(ABC):
    """Abstract base class for all agents."""
    
    @abstractmethod
    def run(self, task: str, context: Optional[Union[str, List[str]]] = None) -> Any:
        """Run the agent on a specific task."""
        pass

class RLMAgent(Agent):
    """Base class for agents powered by RLM."""
    
    def __init__(
        self,
        name: str,
        config: Optional[RLMConfig] = None,
        instructions: Optional[str] = None
    ):
        self.name = name
        self.config = config or RLMConfig()
        self.instructions = instructions
        self.rlm = RLM(config=self.config)
        self.tools = {}
        self.extra_globals = {}

    def add_tool(self, name: str, func: callable):
        """Add a custom tool to the agent's environment."""
        self.tools[name] = func

    def run(self, task: str, context: Optional[Union[str, List[str]]] = None) -> RLMResult:
        """
        Run the agent using RLM.
        
        The task and instructions are combined into the query.
        """
        full_query = f"{self.instructions}\n\nTask: {task}" if self.instructions else task
        return self.rlm.query(
            query=full_query,
            context=context or "No context provided.",
            tools=self.tools,
            extra_globals=self.extra_globals
        )
