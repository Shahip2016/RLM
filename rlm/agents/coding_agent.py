import os
from typing import Optional, List, Union, Dict, Any
from .base import RLMAgent
from rlm import RLMConfig

class CodingAgent(RLMAgent):
    """
    An agent specialized for software engineering tasks.
    
    It has access to filesystem tools like read_file, write_file, 
    and list_dir in its REPL environment.
    """
    
    DEFAULT_INSTRUCTIONS = (
        "You are an expert Software Engineer. You have access to the filesystem "
        "through the following tools:\n"
        "1. list_dir(path: str) -> List[str]\n"
        "2. read_file(path: str) -> str\n"
        "3. write_file(path: str, content: str)\n\n"
        "Use these tools to explore the codebase, read requirements, "
        "and implement or fix code. Always verify your changes."
    )
    
    def __init__(
        self,
        name: str = "CodeBot",
        config: Optional[RLMConfig] = None,
        instructions: Optional[str] = None,
        workspace_root: str = "."
    ):
        super().__init__(
            name=name,
            config=config,
            instructions=instructions or self.DEFAULT_INSTRUCTIONS
        )
        self.workspace_root = workspace_root
        
        # Inject filesystem tools
        self.add_tool("list_dir", self._list_dir)
        self.add_tool("read_file", self._read_file)
        self.add_tool("write_file", self._write_file)
        
    def _list_dir(self, path: str = ".") -> List[str]:
        """List contents of a directory."""
        full_path = os.path.join(self.workspace_root, path)
        try:
            return os.listdir(full_path)
        except Exception as e:
            return [f"Error: {str(e)}"]

    def _read_file(self, path: str) -> str:
        """Read content from a file."""
        full_path = os.path.join(self.workspace_root, path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error: {str(e)}"

    def _write_file(self, path: str, content: str):
        """Write content to a file."""
        full_path = os.path.join(self.workspace_root, path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error: {str(e)}"
