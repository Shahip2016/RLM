# RLM (Recursive Language Model)

A implementation of **Recursive Language Models** (RLM) with a modern Web UI and support for proprietary models (OpenAI, Anthropic, Gemini).

RLM enables LLMs to process **arbitrarily long prompts** by treating them as part of an external environment rather than feeding them directly into the model. The key insight is that long contexts should be symbolically manipulated through code execution and recursive sub-calls.

## 🚀 Features

- **Modern Web UI**: Built with React, Vite, Tailwind CSS, and Framer Motion.
- **Recursive Reasoning**: Automated iterative REPL loop for deep context analysis.
- **Multi-Model Support**: Integrated with OpenAI, Anthropic (Claude-3.5), and Google (Gemini-1.5).
- **Trajectory Visualizer**: Inspect the step-by-step reasoning process and internal REPL states.
- **Context Reservoir**: Support for multiple file uploads and remote URL context.
- **Cost Tracking**: Automatic token counting and cost estimation across all providers.

## 🛠️ Installation & Setup

### 1. Requirements
Ensure you have Python 3.9+ and Node.js 18+ installed.

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt
pip install fastapi uvicorn anthropic google-generativeai

# Start the FastAPI service
python main.py
```

### 3. Frontend Setup
```bash
cd web_ui
npm install
npm run dev
```

## 📖 Usage

1. Open `http://localhost:5173` in your browser.
2. **Context**: Upload documents (PDF/Text) or add URLs to the Reservoir.
3. **Settings**: Enter your API keys (optional if set in `.env`) and select models.
4. **Query**: Ask a question. Watch RLM brainstorm and execute recursive calls.
5. **Inspect**: Click "Inspect Reasoning" to see the full trajectory and code executed.

## 🏗️ Project Structure

- `rlm/`: Core RLM logic and provider integrations.
  - `llm_client.py`: Multi-provider client (OpenAI, Anthropic, Gemini).
  - `rlm.py`: Main recursive orchestrator.
- `main.py`: FastAPI backend service.
- `web_ui/`: Modern React dashboard.
- `examples/`: Legacy Python examples.

## 📜 License
MIT License
