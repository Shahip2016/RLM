# RLM - Recursive Language Models

A Python implementation of the Recursive Language Models (RLM) framework from the paper "Recursive Language Models" by Zhang, Kraska, and Khattab (MIT CSAIL).

## Overview

RLM enables LLMs to process **arbitrarily long prompts** by treating them as part of an external environment rather than feeding them directly into the model. The key insight is that long contexts should be symbolically manipulated through code execution and recursive sub-calls.

### Key Features

- 🔄 **Recursive sub-calls**: LLMs can invoke themselves on portions of the context
- 💻 **Python REPL**: Context is loaded as a variable that can be programmatically examined
- 📊 **Cost tracking**: Automatic token counting and cost estimation
- 🎯 **Task-agnostic**: Works across different types of long-context tasks

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```

2. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

## Quick Start

```python
from rlm import RLM

# Initialize RLM
rlm = RLM()

# Your long document
document = "..." # Very long text

# Query
result = rlm.query(
    query="What is the main theme of this document?",
    context=document
)

print(result.answer)
print(result.usage_summary)
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                         RLM Flow                            │
├─────────────────────────────────────────────────────────────┤
│  1. User provides query + context                           │
│  2. Context loaded into REPL as 'context' variable          │
│  3. LLM writes code to examine/chunk context                │
│  4. LLM uses llm_query() for recursive sub-calls            │
│  5. Process repeats until FINAL() answer                    │
└─────────────────────────────────────────────────────────────┘
```

The LLM can write code like:

```python
# Chunk and analyze
for i, chunk in enumerate(chunks):
    answer = llm_query(f"Analyze chunk {i}: {chunk}")
    buffers.append(answer)

# Aggregate
final = llm_query(f"Combine findings: {buffers}")
```

## Examples

### Simple Example
```bash
python examples/simple_example.py
```

### Needle-in-a-Haystack
```bash
python examples/niah_example.py
```

## Configuration Options

```python
from rlm import RLM, RLMConfig

config = RLMConfig(
    root_model="gpt-4o",        # Model for root LM
    sub_model="gpt-4o-mini",    # Model for sub-calls
    max_iterations=50,          # Max REPL iterations
    max_output_tokens=16384,    # Max tokens per response
    model_variant="gpt"         # "gpt" or "qwen"
)

rlm = RLM(config=config)
```

## Project Structure

```
rlm/
├── __init__.py          # Package exports
├── config.py            # Configuration management
├── llm_client.py        # OpenAI API wrapper
├── prompts.py           # System prompts from paper
├── repl_environment.py  # Python REPL environment
└── rlm.py               # Main orchestrator

examples/
├── simple_example.py    # Basic usage
└── niah_example.py      # Needle-in-a-haystack
```

## Citation

This implementation is based on:

```
@article{zhang2025rlm,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025}
}
```

## License

MIT License
