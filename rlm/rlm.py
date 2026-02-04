"""
Main RLM (Recursive Language Model) orchestrator.

This module provides the core RLM class that:
- Accepts a query and context
- Manages the REPL environment
- Orchestrates the iterative LLM calls
- Extracts the final answer
"""

import re
from typing import Any, Optional, List, Dict, Union
from dataclasses import dataclass, field

from .config import RLMConfig
from .llm_client import LLMClient
from .repl_environment import REPLEnvironment, REPLResult
from .prompts import get_system_prompt


@dataclass
class RLMResult:
    """Result from an RLM query."""
    answer: str
    success: bool
    iterations: int
    total_cost: float
    usage_summary: str
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class RLM:
    """
    Recursive Language Model implementation.
    
    RLM treats long prompts as part of an external environment,
    allowing the LLM to programmatically examine, decompose, and
    recursively call itself over snippets of the prompt.
    
    Example:
        ```python
        from rlm import RLM
        
        rlm = RLM()
        result = rlm.query(
            query="What is the main theme of this document?",
            context="... very long document text ..."
        )
        print(result.answer)
        ```
    """
    
    def __init__(
        self,
        config: Optional[RLMConfig] = None,
        allow_subcalls: bool = True
    ):
        """
        Initialize the RLM.
        
        Args:
            config: RLM configuration (uses defaults if not provided)
            allow_subcalls: Whether to enable recursive llm_query calls
        """
        self.config = config or RLMConfig()
        self.allow_subcalls = allow_subcalls
        self.llm_client = LLMClient(self.config)
    
    def query(
        self,
        query: str,
        context: Union[str, List[str]],
        verbose: bool = False
    ) -> RLMResult:
        """
        Process a query with the given context using RLM.
        
        Args:
            query: The question or task to answer
            context: The context data (string or list of strings)
            verbose: Whether to print progress information
            
        Returns:
            RLMResult with the answer and metadata
        """
        # Reset usage tracking
        self.llm_client.reset_usage()
        
        # Get context info for the system prompt
        context_info = self._get_context_info(context)
        
        # Build the system prompt
        system_prompt = get_system_prompt(
            context_type=context_info['type'],
            context_total_length=context_info['total_length'],
            context_lengths=context_info['chunk_lengths'],
            model_variant=self.config.model_variant,
            allow_subcalls=self.allow_subcalls
        )
        
        # Create the REPL environment
        repl = REPLEnvironment(
            context=context,
            llm_query_fn=self._create_sub_query_fn(),
            max_output_length=self.config.max_context_display
        )
        
        # Build initial message history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        trajectory = []
        
        # Iterative loop
        for iteration in range(self.config.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get LLM response
            response = self.llm_client.query(
                prompt="",  # Using messages instead
                messages=messages,
                model=self.config.root_model
            )
            
            if verbose:
                print(f"LLM Response:\n{response[:500]}...")
            
            # Record in trajectory
            trajectory.append({
                "iteration": iteration + 1,
                "response": response,
                "type": "llm_response"
            })
            
            # Check for final answer
            final_answer = self._extract_final_answer(response, repl)
            if final_answer is not None:
                return RLMResult(
                    answer=final_answer,
                    success=True,
                    iterations=iteration + 1,
                    total_cost=self.llm_client.get_total_cost(),
                    usage_summary=self.llm_client.get_usage_summary(),
                    trajectory=trajectory
                )
            
            # Extract and execute REPL code
            code_blocks = self._extract_code_blocks(response)
            
            if code_blocks:
                repl_output = ""
                for code in code_blocks:
                    if verbose:
                        print(f"\nExecuting code:\n{code[:300]}...")
                    
                    result = repl.execute(code)
                    
                    trajectory.append({
                        "iteration": iteration + 1,
                        "code": code,
                        "output": result.output,
                        "error": result.error,
                        "type": "repl_execution"
                    })
                    
                    if result.success:
                        repl_output += result.output
                    else:
                        repl_output += f"Error: {result.error}"
                
                if verbose:
                    print(f"\nREPL Output:\n{repl_output[:500]}...")
                
                # Add to message history
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"REPL Output:\n{repl_output}"})
            else:
                # No code blocks - prompt to continue or provide answer
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please continue with your analysis using the REPL, or provide your final answer using FINAL() or FINAL_VAR()."
                })
        
        # Max iterations reached
        return RLMResult(
            answer="",
            success=False,
            iterations=self.config.max_iterations,
            total_cost=self.llm_client.get_total_cost(),
            usage_summary=self.llm_client.get_usage_summary(),
            trajectory=trajectory,
            error="Max iterations reached without finding an answer"
        )
    
    def _create_sub_query_fn(self):
        """Create the llm_query function for the REPL environment."""
        def llm_query(prompt: str) -> str:
            return self.llm_client.query(
                prompt=prompt,
                model=self.config.sub_model,
                max_tokens=self.config.max_output_tokens
            )
        return llm_query
    
    def _get_context_info(self, context: Union[str, List[str]]) -> Dict[str, Any]:
        """Get information about the context for the system prompt."""
        if isinstance(context, str):
            return {
                'type': 'string',
                'total_length': len(context),
                'chunk_lengths': [len(context)]
            }
        elif isinstance(context, list):
            return {
                'type': 'list',
                'total_length': sum(len(str(c)) for c in context),
                'chunk_lengths': [len(str(c)) for c in context]
            }
        else:
            return {
                'type': type(context).__name__,
                'total_length': len(str(context)),
                'chunk_lengths': [len(str(context))]
            }
    
    def _extract_code_blocks(self, response: str) -> List[str]:
        """Extract ```repl code blocks from the response."""
        # Match ```repl ... ``` blocks
        pattern = r'```repl\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        return matches
    
    def _extract_final_answer(
        self,
        response: str,
        repl: REPLEnvironment
    ) -> Optional[str]:
        """
        Extract final answer from response if present.
        
        Looks for:
        - FINAL(answer text)
        - FINAL_VAR(variable_name)
        
        Returns None if no final answer found.
        """
        # Check for FINAL_VAR first (variable reference)
        var_match = re.search(r'FINAL_VAR\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)', response)
        if var_match:
            var_name = var_match.group(1)
            value = repl.get_variable(var_name)
            if value is not None:
                return str(value)
        
        # Check for FINAL(text)
        # Handle both single-line and multi-line
        final_match = re.search(r'FINAL\s*\((.*?)\)(?:\s*$|\s*\n)', response, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()
        
        # Also check for FINAL at the very end of response
        final_match = re.search(r'FINAL\s*\((.+)\)\s*$', response, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()
        
        return None
