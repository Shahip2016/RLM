import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm.agents import ResearchAgent, CodingAgent
from rlm import RLMConfig

def research_demo():
    print("\n" + "="*50)
    print("RESEARCH AGENT DEMO")
    print("="*50)
    
    # Simple context
    context = (
        "Project Aether aims to colonize Mars by 2035 using nuclear thermal rockets. "
        "The project is led by Dr. Elena Vance and has a budget of $500 billion. "
        "Key challenges include radiation shielding and sustainable life support."
    )
    
    agent = ResearchAgent(name="AetherBot")
    task = "Summarize Project Aether, including the leader, timeline, and main challenges."
    
    print(f"Task: {task}")
    print("Processing...")
    
    result = agent.run(task, context=context)
    
    print("\nResearch Result:")
    print(result.answer)

def coding_demo():
    print("\n" + "="*50)
    print("CODING AGENT DEMO")
    print("="*50)
    
    # We'll use the current directory as workspace
    agent = CodingAgent(name="DevBot", workspace_root=".")
    
    # Task to list files and read one
    task = "List the files in the current directory and read the content of 'requirements.txt'."
    
    print(f"Task: {task}")
    print("Processing...")
    
    # Coding agent uses RLM to execute its tools in the REPL
    result = agent.run(task, context="This agent has filesystem access.")
    
    print("\nCoding Result:")
    print(result.answer)

def main():
    # Only run if API keys are set (or it will use mock if RLM supports it, 
    # but here we just want to show the interface)
    print("RLM Agents Demonstration")
    
    try:
        research_demo()
    except Exception as e:
        print(f"Research Demo failed: {e}")
        
    try:
        coding_demo()
    except Exception as e:
        print(f"Coding Demo failed: {e}")

if __name__ == "__main__":
    main()
