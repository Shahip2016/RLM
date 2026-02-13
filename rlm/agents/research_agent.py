from .base import RLMAgent
from typing import Optional, List, Union
from rlm import RLMConfig

class ResearchAgent(RLMAgent):
    """
    An agent specialized for research, fact-finding, and synthesis.
    
    It uses RLM to recursively analyze large amounts of context to
    extract insights or answer complex questions.
    """
    
    DEFAULT_INSTRUCTIONS = (
        "You are an expert Research Assistant. Your goal is to provide "
        "comprehensive, accurate, and well-cited information based on the "
        "provided context. Always look for nuances and contradictory evidence. "
        "Synthesize multiple viewpoints when possible."
    )
    
    def __init__(
        self,
        name: str = "ResearchBot",
        config: Optional[RLMConfig] = None,
        instructions: Optional[str] = None
    ):
        super().__init__(
            name=name,
            config=config,
            instructions=instructions or self.DEFAULT_INSTRUCTIONS
        )
        # Research agents might benefit from more iterations by default
        self.config.max_iterations = max(self.config.max_iterations, 20)
