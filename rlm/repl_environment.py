"""
REPL Environment for the RLM framework.

Provides an isolated Python execution environment with:
- Pre-loaded context variable
- llm_query function for recursive sub-calls
- Output capture and state persistence
"""

import sys
import io
import traceback
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field


@dataclass
class REPLResult:
    """Result from REPL code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)


class REPLEnvironment:
    """
    Isolated Python REPL environment for RLM.
    
    The environment provides:
    - A 'context' variable containing the input data
    - An 'llm_query' function for making recursive LLM calls
    - Captured print() output
    - Persistent state across executions
    """
    
    def __init__(
        self,
        context: Any,
        llm_query_fn: Callable[[str], str],
        max_output_length: int = 5000,
        additional_globals: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the REPL environment.
        
        Args:
            context: The input context (string or list of strings)
            llm_query_fn: Function to call for recursive LLM queries
            max_output_length: Max characters to return from output
            additional_globals: Additional variables or tools to inject
        """
        self.max_output_length = max_output_length
        self._output_buffer: List[str] = []
        self._llm_query_fn = llm_query_fn
        
        # Create the execution namespace
        self.globals: Dict[str, Any] = {
            '__builtins__': __builtins__,
            'context': context,
            'llm_query': self._wrapped_llm_query,
            'print': self._capture_print,
        }
        
        # Inject additional globals if provided
        if additional_globals:
            self.globals.update(additional_globals)
        
        # Import common modules
        self._setup_imports()
    
    def _setup_imports(self):
        """Pre-import commonly used modules."""
        import_code = """
import re
import json
import math
from collections import Counter, defaultdict
"""
        try:
            exec(import_code, self.globals)
        except Exception:
            pass  # Ignore import errors
    
    def _capture_print(self, *args, **kwargs):
        """Capture print statements to the output buffer."""
        output = io.StringIO()
        print(*args, file=output, **kwargs)
        self._output_buffer.append(output.getvalue())
    
    def _wrapped_llm_query(self, prompt: str) -> str:
        """
        Wrapped llm_query that captures calls for logging.
        
        Args:
            prompt: The query to send to the sub-LLM
            
        Returns:
            The sub-LLM's response
        """
        self._capture_print(f"[Calling llm_query with {len(prompt)} chars...]")
        result = self._llm_query_fn(prompt)
        self._capture_print(f"[llm_query returned {len(result)} chars]")
        return result
    
    def execute(self, code: str) -> REPLResult:
        """
        Execute code in the REPL environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            REPLResult with output and any errors
        """
        self._output_buffer = []
        
        try:
            # Execute the code
            exec(code, self.globals)
            
            # Get captured output
            output = "".join(self._output_buffer)
            
            # Truncate if necessary
            if len(output) > self.max_output_length:
                output = (
                    output[:self.max_output_length // 2] +
                    f"\n\n... [truncated {len(output) - self.max_output_length} chars] ...\n\n" +
                    output[-self.max_output_length // 2:]
                )
            
            return REPLResult(
                success=True,
                output=output,
                variables=self._get_user_variables()
            )
            
        except Exception as e:
            # Capture the error
            error_msg = traceback.format_exc()
            output = "".join(self._output_buffer)
            
            return REPLResult(
                success=False,
                output=output,
                error=error_msg,
                variables=self._get_user_variables()
            )
    
    def _get_user_variables(self) -> Dict[str, Any]:
        """Get user-defined variables (excluding builtins and functions)."""
        excluded = {
            '__builtins__', 'context', 'llm_query', 'print',
            're', 'json', 'math', 'Counter', 'defaultdict'
        }
        
        result = {}
        for name, value in self.globals.items():
            if name not in excluded and not name.startswith('_'):
                # Only include serializable types
                if isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                    result[name] = value
                else:
                    result[name] = f"<{type(value).__name__}>"
        
        return result
    
    def get_variable(self, name: str) -> Any:
        """
        Get a variable from the environment.
        
        Args:
            name: Variable name
            
        Returns:
            The variable value or None if not found
        """
        return self.globals.get(name)
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get information about the loaded context."""
        context = self.globals.get('context')
        
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
                'chunk_lengths': [len(str(c)) for c in context],
                'num_chunks': len(context)
            }
        else:
            return {
                'type': type(context).__name__,
                'total_length': len(str(context)),
                'chunk_lengths': [len(str(context))]
            }
