import React, { useState, useEffect, useRef } from 'react';
import { Send, Upload, Settings, ChevronRight, Terminal, PieChart, Layers, Brain, Loader2, Plus, X, FileText, Globe, Code, MessageSquare, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const API_BASE = "http://localhost:8000";

const ModelOptions = {
  gpt: [
    { id: "gpt-4o", name: "GPT-4o (Root)", provider: "openai" },
    { id: "gpt-4o-mini", name: "GPT-4o Mini (Sub)", provider: "openai" },
  ],
  anthropic: [
    { id: "claude-3-5-sonnet", name: "Claude 3.5 Sonnet", provider: "anthropic" },
    { id: "claude-3-haiku", name: "Claude 3 Haiku", provider: "anthropic" },
  ],
  google: [
    { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro", provider: "google" },
    { id: "gemini-1.5-flash", name: "Gemini 1.5 Flash", provider: "google" },
  ]
};

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [contexts, setContexts] = useState([]);
  const [urlInput, setUrlInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [activeTrajectory, setActiveTrajectory] = useState(null);
  const [trajIndex, setTrajIndex] = useState(0);

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
  }, [messages, isQuerying]);

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    setIsUploading(true);
    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const resp = await axios.post(`${API_BASE}/upload`, formData);
        setContexts(prev => [...prev, { name: file.name, content: resp.data.content }]);
        setMessages(prev => [...prev, {
          role: 'system',
          content: `Added ${file.name} to context reservoir.`,
          type: 'info'
        }]);
      } catch (err) {
        setMessages(prev => [...prev, { role: 'system', content: `Upload failed for ${file.name}: ${err.message}`, type: 'error' }]);
      }
    }
    setIsUploading(false);
  };

  const handleUrlAdd = () => {
    if (!urlInput.trim()) return;
    setContexts(prev => [...prev, { name: urlInput, content: `Content from ${urlInput} (Simulated via RLM Proxy)` }]);
    setMessages(prev => [...prev, {
      role: 'system',
      content: `URL context added: ${urlInput}`,
      type: 'info'
    }]);
    setUrlInput("");
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsQuerying(true);
    setActiveTrajectory(null);

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
        setTrajIndex(resp.data.trajectory.length - 1);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.response?.data?.detail || err.message}`, type: 'error' }]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#020617] text-slate-200 overflow-hidden font-sans selection:bg-primary-500/30">

      {/* Sidebar - Context Reservoir */}
      <div className="w-80 glass border-r border-white/5 flex flex-col p-6 gap-6 z-20">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl shadow-lg shadow-primary-500/20">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-lg font-bold tracking-tight text-white">RLM <span className="text-primary-400 font-medium">Studio</span></h1>
        </div>

        <div className="space-y-6 flex-1 overflow-y-auto scrollbar-hide">
          {/* File Upload Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 flex items-center gap-2">
                <FileText className="w-3.5 h-3.5" /> Reservoir
              </h3>
              <span className="text-[10px] bg-white/5 px-2 py-0.5 rounded-full text-slate-400">{contexts.length} items</span>
            </div>

            <label className="flex flex-col items-center justify-center w-full h-24 border border-dashed border-white/10 rounded-xl cursor-pointer hover:border-primary-500/30 hover:bg-primary-500/5 transition-all group">
              <Upload className="w-5 h-5 mb-2 text-slate-500 group-hover:text-primary-400" />
              <p className="text-[10px] text-slate-500 font-medium">Add Context Files</p>
              <input type="file" className="hidden" multiple onChange={handleFileUpload} />
            </label>

            <div className="space-y-2">
              <AnimatePresence>
                {contexts.map((ctx, i) => (
                  <motion.div
                    initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, scale: 0.95 }}
                    key={i}
                    className="flex items-center justify-between p-2.5 bg-white/[0.03] border border-white/[0.05] rounded-lg group hover:bg-white/[0.06] transition-all"
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className="w-1.5 h-1.5 rounded-full bg-primary-500 shrink-0" />
                      <span className="text-[11px] font-medium truncate text-slate-300">{ctx.name}</span>
                    </div>
                    <button onClick={() => setContexts(prev => prev.filter((_, idx) => idx !== i))} className="text-slate-600 hover:text-red-400 p-1 opacity-0 group-hover:opacity-100 transition-all">
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>

          {/* URL Scraper Section */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 flex items-center gap-2">
              <Globe className="w-3.5 h-3.5" /> Remote Node
            </h3>
            <div className="flex gap-2">
              <input
                className="flex-1 bg-white/[0.03] border border-white/[0.05] rounded-lg text-[11px] py-2 px-3 focus:outline-none focus:ring-1 focus:ring-primary-500/50"
                placeholder="https://..."
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
              />
              <button onClick={handleUrlAdd} className="p-2 bg-white/5 hover:bg-white/10 rounded-lg text-slate-400 transition-all">
                <Plus className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Settings Section */}
          <div className="space-y-4 pt-4 border-t border-white/5">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 flex items-center gap-2">
              <Settings className="w-3.5 h-3.5" /> System Config
            </h3>
            <div className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-[10px] font-bold text-slate-600 uppercase ml-1">Root Intelligence</label>
                <select
                  className="w-full bg-white/[0.03] border border-white/[0.05] rounded-lg text-[11px] py-2.5 px-3 focus:outline-none focus:ring-1 focus:ring-primary-500/50 appearance-none"
                  value={config.root_model}
                  onChange={(e) => setConfig({ ...config, root_model: e.target.value })}
                >
                  {Object.entries(ModelOptions).map(([provider, models]) => (
                    <optgroup key={provider} label={provider.toUpperCase()} className="bg-slate-900">
                      {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                    </optgroup>
                  ))}
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] font-bold text-slate-600 uppercase ml-1">Sub-Routine Engine</label>
                <select
                  className="w-full bg-white/[0.03] border border-white/[0.05] rounded-lg text-[11px] py-2.5 px-3 focus:outline-none focus:ring-1 focus:ring-primary-500/50 appearance-none"
                  value={config.sub_model}
                  onChange={(e) => setConfig({ ...config, sub_model: e.target.value })}
                >
                  {Object.entries(ModelOptions).map(([provider, models]) => (
                    <optgroup key={provider} label={provider.toUpperCase()} className="bg-slate-900">
                      {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                    </optgroup>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Orchestration Area */}
      <div className="flex-1 flex flex-col relative bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black overflow-hidden">

        {/* Top Header */}
        <div className="h-16 border-b border-white/5 flex items-center justify-between px-8 bg-slate-950/50 backdrop-blur-md z-10">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1 bg-primary-500/10 border border-primary-500/20 rounded-full">
              <div className="w-1.5 h-1.5 rounded-full bg-primary-500 animate-pulse" />
              <span className="text-[10px] font-bold text-primary-400 tracking-wider">RLM ACTIVE</span>
            </div>
          </div>
          <div className="flex items-center gap-6 text-[11px] font-mono text-slate-500">
            <span className="flex items-center gap-2"><PieChart className="w-3.5 h-3.5" /> $0.0000 Total</span>
            <span className="flex items-center gap-2"><Layers className="w-3.5 h-3.5" /> Latency: 45ms</span>
          </div>
        </div>

        {/* Chat Feed */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-8 py-10 space-y-10 scroll-smooth relative"
        >
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-slate-700 select-none max-w-md mx-auto text-center gap-6">
              <div className="p-6 rounded-3xl bg-white/[0.02] border border-white/[0.04]">
                <Brain className="w-16 h-16 mb-4 text-slate-800" />
                <h2 className="text-xl font-semibold text-slate-500">Recursive Language Model</h2>
                <p className="mt-2 text-sm text-slate-600">Load a context reservoir and initiate a query to begin multi-step recursive reasoning.</p>
              </div>
              <div className="grid grid-cols-2 gap-4 w-full">
                {['Financial Analysis', 'Code Review', 'Paper Summary', 'Deep Research'].map(t => (
                  <button key={t} className="p-3 text-[11px] font-bold border border-white/5 rounded-xl hover:bg-white/5 transition-all uppercase tracking-widest text-slate-600">
                    {t}
                  </button>
                ))}
              </div>
            </div>
          )}

          <AnimatePresence>
            {messages.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start hover-trigger'}`}
              >
                <div className={`group relative max-w-[85%] rounded-3xl transition-all ${m.role === 'user'
                    ? 'bg-gradient-to-br from-primary-600 to-primary-700 p-5 pr-6 text-white shadow-2xl shadow-primary-500/20'
                    : m.type === 'info'
                      ? 'bg-white/[0.02] border border-white/[0.05] p-3 px-4 text-slate-500 text-[11px] font-medium'
                      : 'bg-white/[0.03] border border-white/[0.05] p-6 shadow-xl backdrop-blur-sm'
                  }`}>
                  <div className="flex items-start gap-4">
                    {m.role === 'assistant' && (
                      <div className="shrink-0 mt-1 p-1.5 bg-primary-500/10 rounded-lg border border-primary-500/20">
                        <Brain className="w-4 h-4 text-primary-400" />
                      </div>
                    )}
                    <div className="flex-1 space-y-4">
                      <div className="prose prose-invert prose-sm max-w-none leading-relaxed text-slate-300">
                        {m.content}
                      </div>

                      {m.trajectory && (
                        <div className="flex items-center gap-3 pt-4 border-t border-white/5">
                          <button
                            onClick={() => setActiveTrajectory(m.trajectory)}
                            className="flex items-center gap-2 px-3 py-1.5 bg-primary-500/10 hover:bg-primary-500/20 rounded-full text-[10px] font-bold text-primary-400 border border-primary-500/20 transition-all uppercase tracking-wider"
                          >
                            <Terminal className="w-3 h-3" /> Inspect Reasoning ({m.iterations})
                          </button>
                          <div className="flex gap-4 text-[10px] font-mono text-slate-500 uppercase font-bold tracking-tighter">
                            <span className="flex items-center gap-1.5"><Layers className="w-3 h-3" /> {m.iterations} Iterations</span>
                            <span className="flex items-center gap-1.5 text-emerald-500/70"><PieChart className="w-3 h-3" /> Cost: ${m.cost.toFixed(5)}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {isQuerying && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
              <div className="bg-white/[0.03] border border-white/5 p-6 rounded-3xl flex flex-col gap-4 min-w-[300px]">
                <div className="flex items-center gap-3">
                  <div className="p-1.5 bg-primary-500/20 rounded-lg">
                    <Loader2 className="w-4 h-4 animate-spin text-primary-400" />
                  </div>
                  <span className="text-sm font-bold tracking-tight text-slate-400 uppercase tracking-widest text-[11px]">Reasoning Chain Active</span>
                </div>
                <div className="h-1.5 w-full bg-white/[0.05] rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-primary-500"
                    animate={{ x: ["-100%", "100%"] }}
                    transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Floating Input Controller */}
        <div className="w-full max-w-4xl mx-auto pb-10 px-8">
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-primary-600 to-indigo-600 rounded-3xl blur opacity-20 group-focus-within:opacity-40 transition-all duration-500" />
            <div className="relative flex items-center gap-3 bg-slate-900/80 backdrop-blur-xl border border-white/10 p-3 pl-5 rounded-2xl shadow-2xl">
              <div className="p-2 bg-white/5 rounded-xl text-slate-500">
                <MessageSquare className="w-4 h-4" />
              </div>
              <input
                className="flex-1 bg-transparent border-none focus:ring-0 text-white placeholder-slate-600 text-[13px] font-medium"
                placeholder="Send a recursive query to the orchestration engine..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              />
              <button
                onClick={handleSend}
                disabled={isQuerying || !input.trim() || contexts.length === 0}
                className="p-3 px-6 bg-primary-500 hover:bg-primary-400 disabled:opacity-30 disabled:grayscale text-white rounded-xl transition-all shadow-xl shadow-primary-500/10 flex items-center gap-2 group/btn"
              >
                <span className="text-[11px] font-black uppercase tracking-widest ml-1">Execute</span>
                <ChevronRight className="w-4 h-4 group-hover/btn:translate-x-0.5 transition-transform" />
              </button>
            </div>
            {contexts.length === 0 && (
              <p className="absolute -bottom-6 left-5 text-[10px] font-bold text-red-400/60 uppercase tracking-widest flex items-center gap-1.5">
                <Info className="w-3 h-3" /> System idle: Missing context reservoir
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Trajectory Inspector Overlay */}
      <AnimatePresence>
        {activeTrajectory && (
          <>
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              onClick={() => setActiveTrajectory(null)}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm z-40"
            />
            <motion.div
              initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="absolute right-0 inset-y-0 w-[500px] bg-[#020617]/95 backdrop-blur-2xl border-l border-white/10 z-50 flex flex-col shadow-[0_0_100px_-20px_rgba(0,0,0,0.8)]"
            >
              <div className="p-8 border-b border-white/5 flex items-center justify-between">
                <div className="flex flex-col gap-1">
                  <h2 className="text-sm font-black uppercase tracking-[0.2em] text-primary-400">Trajectory Explorer</h2>
                  <p className="text-[10px] text-slate-500 font-mono uppercase tracking-widest">Internal state visualization</p>
                </div>
                <button onClick={() => setActiveTrajectory(null)} className="p-2 hover:bg-white/5 rounded-xl text-slate-500 hover:text-white transition-all">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex-1 overflow-hidden flex flex-col">
                {/* Step Timeline */}
                <div className="flex gap-2 p-6 overflow-x-auto scrollbar-hide border-b border-white/5 bg-black/20">
                  {activeTrajectory.map((step, idx) => (
                    <button
                      key={idx}
                      onClick={() => setTrajIndex(idx)}
                      className={`shrink-0 flex flex-col items-center gap-2 p-3 rounded-2xl border transition-all ${trajIndex === idx ? 'bg-primary-500/20 border-primary-500 shadow-lg shadow-primary-500/10' : 'bg-white/5 border-transparent opacity-60'
                        }`}
                    >
                      <span className="text-[10px] font-black">{idx + 1}</span>
                      <div className={`p-1.5 rounded-lg ${step.type === 'repl_execution' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-primary-500/20 text-primary-400'}`}>
                        {step.type === 'repl_execution' ? <Code className="w-4 h-4" /> : <Brain className="w-4 h-4" />}
                      </div>
                    </button>
                  ))}
                </div>

                {/* Step Content Area */}
                <div className="flex-1 overflow-y-auto p-8 space-y-8 custom-scrollbar">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-black uppercase tracking-widest text-slate-500">Step {trajIndex + 1} Metadata</span>
                      <span className={`px-2 py-1 rounded-md text-[9px] font-black uppercase tracking-tighter ${activeTrajectory[trajIndex].type === 'repl_execution' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                        }`}>
                        {activeTrajectory[trajIndex].type}
                      </span>
                    </div>

                    <div className="space-y-6">
                      {activeTrajectory[trajIndex].code && (
                        <div className="space-y-3">
                          <label className="text-[10px] font-bold text-slate-600 uppercase tracking-widest flex items-center gap-2">
                            <Code className="w-3.5 h-3.5" /> Executed Logic
                          </label>
                          <div className="bg-black border border-white/5 rounded-2xl p-6 font-mono text-[12px] leading-relaxed text-emerald-400 shadow-inner overflow-x-auto whitespace-pre">
                            {activeTrajectory[trajIndex].code}
                          </div>
                        </div>
                      )}

                      {activeTrajectory[trajIndex].response && (
                        <div className="space-y-3">
                          <label className="text-[10px] font-bold text-slate-600 uppercase tracking-widest flex items-center gap-2">
                            <MessageSquare className="w-3.5 h-3.5" /> Engine Response
                          </label>
                          <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6 text-[12px] leading-loose text-slate-300 italic shadow-md">
                            {activeTrajectory[trajIndex].response}
                          </div>
                        </div>
                      )}

                      {activeTrajectory[trajIndex].output && (
                        <div className="space-y-3">
                          <label className="text-[10px] font-bold text-slate-600 uppercase tracking-widest flex items-center gap-2">
                            <Terminal className="w-3.5 h-3.5" /> Runtime Output
                          </label>
                          <div className="bg-[#0c1117] border-l-4 border-primary-500 rounded-r-2xl p-6 font-mono text-[11px] leading-relaxed text-slate-400 shadow-inner">
                            {activeTrajectory[trajIndex].output}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      <style dangerouslySetInnerHTML={{
        __html: `
        .glass { background: rgba(15, 23, 42, 0.4); backdrop-filter: blur(20px); }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 10px; }
        .hover-trigger:hover .opacity-0 { opacity: 1; }
      `}} />
    </div>
  );
}

export default App;
