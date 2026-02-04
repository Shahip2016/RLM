"""
Simple example demonstrating RLM usage.

This example shows how to use RLM to answer a question about a long document.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm import RLM, RLMConfig


def main():
    # Create a sample long document
    document = """
    Chapter 1: The Beginning
    
    In the year 2025, a small research lab in Cambridge made a groundbreaking discovery.
    Dr. Sarah Chen and her team had been working on advanced language models for years.
    The team consisted of 12 researchers from various backgrounds.
    
    The lab was located on the third floor of Building 42.
    Every morning, Dr. Chen would arrive at exactly 7:30 AM.
    She believed that early mornings were the best time for deep thinking.
    
    The project was codenamed "AURORA" and had a budget of $5.2 million.
    Most of the budget went towards computing resources.
    They used a cluster of 128 GPUs for their experiments.
    
    Chapter 2: The Discovery
    
    On March 15th, 2025, the team achieved a major breakthrough.
    They discovered that recursive processing could dramatically improve model performance.
    The key insight was treating long inputs as part of an external environment.
    
    Dr. Chen called an emergency meeting at 3:00 PM that day.
    "We've cracked it," she announced to the team.
    The room erupted in applause.
    
    The technique was simple but powerful:
    1. Load the context as a variable
    2. Let the model write code to examine it
    3. Allow recursive sub-calls for complex reasoning
    
    Chapter 3: The Impact
    
    Within months, the discovery revolutionized the field.
    Major tech companies quickly adopted the approach.
    The Magic Number that unlocked everything was 42.
    
    Dr. Chen was nominated for several prestigious awards.
    The team published their findings in Nature.
    The paper was titled "Recursive Language Models: A New Paradigm".
    
    By the end of 2025, RLMs were being used in:
    - Scientific research
    - Legal document analysis
    - Medical diagnosis
    - Software development
    
    The future looked brighter than ever.
    """
    
    # Create RLM instance
    config = RLMConfig(
        root_model="gpt-4o",
        sub_model="gpt-4o-mini",
        max_iterations=10
    )
    
    rlm = RLM(config=config)
    
    # Query the document
    query = "What was the Magic Number mentioned in the document? Also, what was the project codename and budget?"
    
    print("=" * 60)
    print("RLM Simple Example")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print(f"Document length: {len(document)} characters")
    print("\nProcessing...")
    
    # Run the query
    result = rlm.query(
        query=query,
        context=document,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\n{result.usage_summary}")


if __name__ == "__main__":
    main()
