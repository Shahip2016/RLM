"""
RLM - Recursive Language Models

A framework for processing arbitrarily long prompts by treating them as 
part of an external environment that the LLM can programmatically interact with.

Based on the paper: "Recursive Language Models" (Zhang, Kraska, Khattab - MIT CSAIL)
"""

from .rlm import RLM
from .config import RLMConfig
from .llm_client import LLMClient
from .repl_environment import REPLEnvironment

__version__ = "0.1.0"
__all__ = ["RLM", "RLMConfig", "LLMClient", "REPLEnvironment"]
