# Recursive Language Models: The "Out-of-Core" Trick That Could Fix Context Windows Forever

*Why feeding 10 million tokens directly to your LLM is probably the wrong approach*

---

I've been thinking a lot about context windows lately. Not the "how do we make them bigger" problem—we've been throwing compute at that for years. I mean the more fundamental question: **should we even be stuffing all that context into the neural network in the first place?**

A new paper from MIT (Zhang, Kraska, and Khattab) gave me a lot to chew on. The idea is elegantly simple, almost embarrassingly so, and yet it dramatically outperforms every baseline they tested against. Let me walk you through it.

## The Problem (That We Keep Ignoring)

Here's what we know but don't talk about enough: modern LLMs exhibit "context rot." Even GPT-5 (yes, they got early access) degrades significantly as context gets longer. The degradation isn't just about hitting token limits—it happens *within* the context window too.

The kicker? This degradation is **task-dependent**. Finding a single needle in a haystack? Models are okay. Aggregating information across the entire input? Performance falls off a cliff.

The authors plot this beautifully. On a quadratic-complexity task (OOLONG-Pairs), GPT-5 scores... 0.04%. Not 4%. Zero-point-zero-four percent. That's essentially random.

## The Core Insight (And Why It's Brilliant)

Here's the idea that made me sit up:

> Long prompts should not be fed into the neural network directly but should instead be treated as part of the environment that the LLM can symbolically interact with.

Stop and think about that. We've been approaching long context like it's a "bigger pipe" problem. The RLM approach says: no, it's an **information retrieval and processing** problem. The LLM should be able to *programmatically examine* the input, not just passively consume it.

This is basically out-of-core algorithms for LLMs. If you've worked with databases or big data systems, you know the trick: when your data doesn't fit in memory, you don't buy more RAM—you design clever algorithms that fetch data on demand.

## How RLMs Work (The Implementation)

The mechanics are surprisingly clean:

1. **Load context into a REPL**: The input becomes a Python variable
2. **LLM writes code**: It can slice, filter, regex-search the context  
3. **Recursive sub-calls**: `llm_query()` spawns sub-LMs on chunks
4. **Build up answer**: Use variables as buffers, aggregate results

Here's what a typical RLM trajectory looks like:

```python
# RLM examining a 10M token corpus
chunks = [context[i:i+100000] for i in range(0, len(context), 100000)]
answers = []
for chunk in chunks:
    ans = llm_query(f"Find evidence about {query} in: {chunk}")
    answers.append(ans)
    
final = llm_query(f"Synthesize findings: {answers}")
```

The model decides *how* to chunk, *what* to search for, and *when* to recurse. No human-engineered pipeline. It just... figures it out.

## The Results (And Why They Surprised Me)

On BrowseComp+ (a benchmark requiring reasoning over 6-11M tokens across 1000 documents):

| Method | Accuracy | Cost |
|--------|----------|------|
| GPT-5 (base) | 0%* | N/A |
| Summary Agent | 70% | $0.57 |
| CodeAct + BM25 | 51% | $0.71 |
| **RLM(GPT-5)** | **91%** | $0.99 |

*couldn't even run—context too long

That's a 30+ point improvement over the summarization approach. And the RLM is *cheaper* than the theoretical cost of GPT-5-mini ingesting those tokens directly (~$1.50-2.75).

But here's what really got me: on OOLONG-Pairs, where base GPT-5 scored 0.04%, **RLM scored 58%**. That's not an incremental improvement. That's going from "completely broken" to "actually useful."

## What the Models Learn to Do (Without Being Told)

The paper has some fascinating analysis of emergent behaviors:

**Filtering with priors**: The model uses regex to search for keywords it has a prior about. Looking for a festival in La Union? It searches for both.

**Smart chunking**: Not uniform splits, but structural boundaries (headers, sections).

**Answer verification**: Sub-calls to double-check answers before returning them.

**Variable-based outputs**: For long-output tasks, results are built up in Python variables and returned at the end.

None of this was explicitly programmed. The base model, given the REPL interface and permission to recurse, *discovers* these strategies.

## My Takes

**This feels like the right abstraction.** I've been skeptical of the "just make context bigger" race. The RLM framing—context as environment, not payload—maps better to how reasoning actually works.

**The prompting is minimal.** They use essentially the same system prompt across all tasks. No task-specific engineering. That's a good sign for generalization.

**Different models, different behaviors.** GPT-5 is conservative with sub-calls. Qwen3-Coder spawns thousands. They had to add a "don't make too many calls" warning for Qwen. This suggests post-training for RLM behavior could yield big gains.

**The depth=1 recursion limit is suspicious.** They only let sub-calls spawn LMs, not sub-RLMs. The paper acknowledges this but doesn't explore it. I suspect deeper recursion could help, with the right guardrails.

**This is inference-time compute scaling, but different.** We talk about test-time compute as "let the model think longer." RLMs are "let the model *access more information* selectively." Both matter.

## The Obvious Next Step

Train models to be RLMs.

Right now they're using off-the-shelf frontier models. The paper hints that RLM trajectories could be viewed as reasoning traces, trainable via STAR-style bootstrapping.

Imagine a model that's been explicitly trained to:
- Recognize when to offload to REPL
- Make efficient chunking decisions  
- Balance sub-call cost vs. accuracy
- Know when it has enough information to answer

The current approach relies on emergent behavior. Purpose-built RLMs could be dramatically more efficient.

## The Takeaway

RLMs aren't magic. They're a formalization of something we should have been thinking about more carefully: the LLM's relationship to its context.

The neural network is a reasoning engine. The context is a data source. Treating them differently—letting the reasoner *query* the data rather than consume it wholesale—turns out to matter a lot.

Context windows will keep growing. But maybe the bigger question isn't "how many tokens can we fit?" but "how intelligently can we access them?"

---

*If you found this interesting, I implemented the full RLM framework in Python based on the paper. The code is [here](link). Happy to discuss in the comments.*

---

**References**
- Zhang, Kraska, Khattab. "Recursive Language Models" (2025). arXiv:2512.24601
