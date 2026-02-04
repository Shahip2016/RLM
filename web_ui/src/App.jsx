import React, { useState, useEffect, useRef } from 'react';
import { Send, Upload, Settings, ChevronRight, Terminal, PieChart, Layers, Brain, Loader2, Plus } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const API_BASE = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [contexts, setContexts] = useState([]); // Changed to list
  const [urlInput, setUrlInput] = useState("");
  const [isScraping, setIsScraping] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [activeTrajectory, setActiveTrajectory] = useState(null);
  const [showSettings, setShowSettings] = useState(false);

  const [config, setConfig] = useState({
    root_model: "gpt-4o",
    sub_model: "gpt-4o-mini",
    max_iterations: 10,
    max_recursion_depth: 1
  });

  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await axios.post(`${API_BASE}/upload`, formData);
      setContext(resp.data.content);
      setMessages(prev => [...prev, {
        role: 'system',
        content: `Uploaded ${file.filename} (${resp.data.length} characters)`,
        type: 'info'
      }]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'system', content: `Upload failed: ${err.message}`, type: 'error' }]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsQuerying(true);

    const fullContext = contexts.map(c => c.content).join("\n\n---\n\n");

    try {
      const resp = await axios.post(`${API_BASE}/query`, {
        query: input,
        context: fullContext,
        config: config
      });

      const assistantMessage = {
        role: 'assistant',
        content: resp.data.answer,
        success: resp.data.success,
        iterations: resp.data.iterations,
        cost: resp.data.total_cost,
        trajectory: resp.data.trajectory
      };

      setMessages(prev => [...prev, assistantMessage]);
      if (resp.data.trajectory) {
        setActiveTrajectory(resp.data.trajectory);
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.response?.data?.detail || err.message}`, type: 'error' }]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 overflow-hidden font-sans">
      {/* Sidebar - Context & Settings */}
      <div className="w-80 glass border-r flex flex-col p-6 gap-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary-600 rounded-lg shadow-lg shadow-primary-500/20">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-xl font-bold tracking-tight">RLM <span className="text-primary-400 font-medium">Core</span></h1>
        </div>

        <div className="space-y-4 flex-1 overflow-y-auto pr-2">
          {/* Context Upload */}
          <div className="glass-card !p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Upload className="w-4 h-4 text-primary-400" /> Context Source
            </h3>
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-white/10 rounded-xl cursor-pointer hover:border-primary-500/50 hover:bg-white/5 transition-all">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <Upload className="w-8 h-8 mb-3 text-slate-400" />
                <p className="text-xs text-slate-400 text-center">Drag files here or click to browse</p>
              </div>
              <input type="file" className="hidden" multiple onChange={handleFileUpload} />
            </label>

            <div className="mt-4 space-y-2 max-h-40 overflow-y-auto">
              {contexts.map((ctx, i) => (
                <div key={i} className="flex items-center justify-between p-2 bg-white/5 rounded-lg border border-white/5 group">
                  <div className="flex items-center gap-2 min-w-0">
                    <div className="w-1 h-1 rounded-full bg-primary-500" />
                    <span className="text-[10px] font-medium truncate text-slate-300">{ctx.name}</span>
                  </div>
                  <button
                    onClick={() => setContexts(prev => prev.filter((_, idx) => idx !== i))}
                    className="text-slate-600 hover:text-red-400 transition-colors"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* URL Input */}
          <div className="glass-card !p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-slate-200">
              <Terminal className="w-4 h-4 text-emerald-400" /> Remote URL
            </h3>
            <div className="relative">
              <input
                className="w-full bg-slate-800 border border-white/5 rounded-lg text-[11px] py-2 px-3 focus:ring-1 focus:ring-primary-500 pr-10"
                placeholder="https://example.com/context"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
              />
              <button
                className="absolute right-2 top-1.5 text-slate-500 hover:text-white"
                onClick={() => {
                  if (!urlInput) return;
                  setContexts(prev => [...prev, { name: urlInput, content: `Content from ${urlInput} (Simulated)` }]);
                  setUrlInput("");
                }}
              >
                <PlusIcon className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Model Config */}
          <div className="glass-card !p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Settings className="w-4 h-4 text-primary-400" /> Options
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1 block">Root Model</label>
                <select
                  className="w-full bg-slate-800 border-none rounded-lg text-xs py-2 px-3 focus:ring-1 focus:ring-primary-500"
                  value={config.root_model}
                  onChange={(e) => setConfig({ ...config, root_model: e.target.value })}
                >
                  <option value="gpt-4o">GPT-4o</option>
                  <option value="claude-3-5-sonnet">Claude 3.5 Sonnet</option>
                  <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                </select>
              </div>
              <div>
                <label className="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1 block">Sub Model</label>
                <select
                  className="w-full bg-slate-800 border-none rounded-lg text-xs py-2 px-3 focus:ring-1 focus:ring-primary-500"
                  value={config.sub_model}
                  onChange={(e) => setConfig({ ...config, sub_model: e.target.value })}
                >
                  <option value="gpt-4o-mini">GPT-4o Mini</option>
                  <option value="claude-3-haiku">Claude 3 Haiku</option>
                  <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <div className="text-[10px] text-slate-500 text-center">
          Recursive Language Model v1.0
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col items-center justify-center relative bg-[radial-gradient(circle_at_50%_50%,#0f172a_0%,#020617_100%)]">
        <div
          ref={scrollRef}
          className="w-full max-w-4xl flex-1 overflow-y-auto px-6 py-10 space-y-8 scroll-smooth"
        >
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center opacity-30 select-none">
              <Brain className="w-24 h-24 mb-6" />
              <p className="text-lg">Paste some context and start a reasoning chain.</p>
            </div>
          )}

          <AnimatePresence>
            {messages.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[85%] rounded-2xl p-5 ${m.role === 'user'
                  ? 'bg-primary-600 text-white shadow-xl shadow-primary-500/20'
                  : m.type === 'info'
                    ? 'bg-white/5 border border-white/10 text-slate-400 text-sm italic'
                    : 'glass shadow-xl'
                  }`}>
                  <div className="flex items-start gap-4">
                    {m.role === 'assistant' && (
                      <div className="mt-1 p-1 bg-primary-500/20 rounded text-primary-400">
                        <Brain className="w-4 h-4" />
                      </div>
                    )}
                    <div className="prose prose-invert prose-sm max-w-none">
                      {m.content}
                    </div>
                  </div>

                  {m.cost !== undefined && (
                    <div className="mt-4 pt-4 border-t border-white/5 flex gap-4 text-[10px] font-mono text-slate-500">
                      <span className="flex items-center gap-1"><Layers className="w-3 h-3" /> {m.iterations} steps</span>
                      <span className="flex items-center gap-1"><PieChart className="w-3 h-3" /> ${m.cost.toFixed(4)}</span>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {isQuerying && (
            <div className="flex justify-start">
              <div className="glass p-5 rounded-2xl flex items-center gap-3 text-slate-400">
                <Loader2 className="w-5 h-5 animate-spin text-primary-500" />
                <span className="text-sm font-medium animate-pulse">Recursive Reasoning in Progress...</span>
              </div>
            </div>
          )}
        </div>

        {/* Input Dock */}
        <div className="w-full max-w-4xl pb-10 px-6">
          <div className="p-2 glass rounded-2xl flex items-center gap-2 group focus-within:ring-2 focus-within:ring-primary-500/40 transition-all">
            <input
              className="flex-1 bg-transparent border-none focus:ring-0 px-4 py-3 placeholder-slate-500 text-sm"
              placeholder="Ask anything about the provided context..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <button
              onClick={handleSend}
              disabled={isQuerying || !input.trim()}
              className="p-3 bg-primary-600 hover:bg-primary-500 disabled:opacity-50 disabled:hover:bg-primary-600 text-white rounded-xl transition-colors shadow-lg shadow-primary-500/20"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Trajectory Panel - Dynamic Overlay */}
      <AnimatePresence>
        {activeTrajectory && (
          <motion.div
            initial={{ x: 400 }}
            animate={{ x: 0 }}
            exit={{ x: 400 }}
            className="w-96 glass border-l absolute right-0 inset-y-0 z-50 p-6 flex flex-col shadow-2xl"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-sm font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
                <Terminal className="w-4 h-4" /> Reasoning Chain
              </h2>
              <button onClick={() => setActiveTrajectory(null)} className="text-slate-500 hover:text-white transition-colors">
                <ChevronRight className="w-6 h-6" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-6 pr-2">
              {activeTrajectory.map((step, idx) => (
                <div key={idx} className="space-y-3">
                  <div className="flex items-center gap-2 text-[10px] font-bold text-primary-400 uppercase tracking-tighter">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary-500" />
                    Iteration {step.iteration} - {step.type}
                  </div>
                  {step.code && (
                    <div className="bg-black/40 rounded-lg p-3 text-[11px] font-mono text-emerald-400 border border-emerald-500/20 overflow-x-auto">
                      {step.code}
                    </div>
                  )}
                  {step.response && (
                    <div className="text-xs text-slate-300 bg-white/5 rounded-lg p-3 leading-relaxed border border-white/5">
                      {step.response.substring(0, 300)}...
                    </div>
                  )}
                  {step.output && (
                    <div className="bg-slate-800 text-[10px] font-mono p-2 rounded border-l-2 border-primary-500 flex flex-col gap-1">
                      <span className="text-slate-500 uppercase tracking-widest text-[8px] font-bold">REPL Output:</span>
                      <span className="text-slate-200">{step.output.substring(0, 500)}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
