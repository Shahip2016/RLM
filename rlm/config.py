"""
Configuration management for the RLM framework.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


@dataclass
class RLMConfig:
    """Configuration for the RLM framework."""
    
    # API Configuration
    api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Model Configuration
    root_model: str = "gpt-4o"  # Model for the root LM
    sub_model: str = "gpt-4o-mini"  # Model for recursive sub-calls
    
    # Token Limits
    max_output_tokens: int = 16384
    max_context_display: int = 5000  # Max chars to show in REPL output
    
    # Iteration Limits
    max_iterations: int = 50  # Max REPL iterations before forcing answer
    max_recursion_depth: int = 1  # Depth of recursive sub-calls (paper uses 1)
    
    # Cost Tracking
    track_costs: bool = True
    
    # Model-specific adjustments
    model_variant: str = "gpt"  # "gpt" or "qwen" - affects system prompt
    
    def __post_init__(self):
        """Load configuration from environment if not provided."""
        load_dotenv()
        
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL")
        
        # Allow environment overrides for models
        env_root = os.getenv("RLM_ROOT_MODEL")
        env_sub = os.getenv("RLM_SUB_MODEL")
        if env_root:
            self.root_model = env_root
        if env_sub:
            self.sub_model = env_sub
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key to RLMConfig."
            )


# Default configuration instance
default_config = RLMConfig()
