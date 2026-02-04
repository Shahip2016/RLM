"""
LLM Client for the RLM framework.

Provides a unified interface to OpenAI-compatible APIs with cost tracking.
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from openai import OpenAI
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .config import RLMConfig


@dataclass
class TokenUsage:
    """Track token usage and costs."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Approximate costs per 1M tokens (as of 2024)
    COSTS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }
    
    def add(self, prompt: int, completion: int):
        """Add token counts."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
    
    def estimate_cost(self, model: str) -> float:
        """Estimate cost in USD based on model."""
        # Find matching model or use default
        model_key = None
        for key in self.COSTS:
            if key in model.lower():
                model_key = key
                break
        
        if model_key is None:
            # Default to gpt-4o pricing
            model_key = "gpt-4o"
        
        costs = self.COSTS[model_key]
        input_cost = (self.prompt_tokens / 1_000_000) * costs["input"]
        output_cost = (self.completion_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost


class LLMClient:
    """
    Client for making LLM API calls with cost tracking.
    
    Supports OpenAI and compatible APIs.
    """
    
    def __init__(self, config: Optional[RLMConfig] = None):
        """Initialize the LLM client."""
        self.config = config or RLMConfig()
        self.config.validate()
        
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Anthropic client
        self.anthropic_client = None
        if ANTHROPIC_AVAILABLE and hasattr(self.config, 'anthropic_api_key') and self.config.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        
        # Gemini setup
        if GEMINI_AVAILABLE and hasattr(self.config, 'gemini_api_key') and self.config.gemini_api_key:
            genai.configure(api_key=self.config.gemini_api_key)
        
        # Token tracking per model
        self.usage: Dict[str, TokenUsage] = {}
        
        # Initialize tokenizer if available
        self._tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
            except Exception:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Rough estimate: ~4 chars per token
        return len(text) // 4
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Make an LLM query.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            model: Model to use (defaults to config.root_model)
            messages: Optional full message history (overrides prompt/system_prompt)
            max_tokens: Max output tokens
            temperature: Sampling temperature
            
        Returns:
            The model's response text
        """
        model = model or self.config.root_model
        max_tokens = max_tokens or self.config.max_output_tokens
        
        # Build messages
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        
        # Handle Anthropic models
        if "claude" in model.lower():
            if not self.anthropic_client:
                raise ValueError(f"Anthropic client not initialized. Check ANTHROPIC_API_KEY for model {model}")
            
            # Convert messages to Anthropic format
            anth_system = ""
            anth_messages = []
            for m in messages:
                if m["role"] == "system":
                    anth_system = m["content"]
                else:
                    anth_messages.append({"role": m["role"], "content": m["content"]})
            
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=anth_system,
                messages=anth_messages
            )
            
            # Track usage
            if response.usage:
                if model not in self.usage:
                    self.usage[model] = TokenUsage()
                self.usage[model].add(
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )
            
            # Concatenate response content
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
            return content

        # Handle Gemini models
        if "gemini" in model.lower():
            if not GEMINI_AVAILABLE:
                raise ValueError(f"Gemini support not available. Install google-generativeai for model {model}")
            
            gemini_model = genai.GenerativeModel(model)
            
            # Format history for Gemini
            gemini_history = []
            for m in messages[:-1]:
                role = "user" if m["role"] in ["user", "system"] else "model"
                gemini_history.append({"role": role, "parts": [m["content"]]})
            
            chat = gemini_model.start_chat(history=gemini_history)
            response = chat.send_message(
                messages[-1]["content"],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            
            # Track usage (Gemini usage info is in response.usage_metadata)
            if hasattr(response, 'usage_metadata'):
                if model not in self.usage:
                    self.usage[model] = TokenUsage()
                self.usage[model].add(
                    response.usage_metadata.prompt_token_count,
                    response.usage_metadata.candidates_token_count
                )
            
            return response.text

        # Make API call (OpenAI)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Track usage
        if response.usage:
            if model not in self.usage:
                self.usage[model] = TokenUsage()
            self.usage[model].add(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        
        return response.choices[0].message.content or ""
    
    def get_total_cost(self) -> float:
        """Get total estimated cost across all models."""
        total = 0.0
        for model, usage in self.usage.items():
            total += usage.estimate_cost(model)
        return total
    
    def get_usage_summary(self) -> str:
        """Get a summary of token usage and costs."""
        lines = ["Token Usage Summary:"]
        total_cost = 0.0
        
        for model, usage in self.usage.items():
            cost = usage.estimate_cost(model)
            total_cost += cost
            lines.append(
                f"  {model}: {usage.prompt_tokens:,} in / {usage.completion_tokens:,} out "
                f"(${cost:.4f})"
            )
        
        lines.append(f"  Total: ${total_cost:.4f}")
        return "\n".join(lines)
    
    def reset_usage(self):
        """Reset usage tracking."""
        self.usage = {}
