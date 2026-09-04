'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAriaStore } from '../store/useAriaStore';
import { useAriaWebSocket } from '../hooks/useAriaWebSocket';

export const TerminalFeed = () => {
  const { messages, connectionState } = useAriaStore();
  const { sendMessage } = useAriaWebSocket();
  const feedRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [inputValue, setInputValue] = useState('');
  const [isListening, setIsListening] = useState(false);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
  }, [messages]);

  const handleSend = () => {
    if (inputValue.trim()) {
      sendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  const toggleListen = () => {
    setIsListening(!isListening);
    // Voice recognition logic will go here
  };

  return (
    <div className="flex flex-col h-full w-full max-h-screen backdrop-blur-md bg-black/80 border border-[#00f0ff]/40 rounded-xl shadow-[0_0_30px_rgba(0,240,255,0.15)] overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#00f0ff]/20 bg-slate-950/50">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full animate-pulse bg-[#00f0ff] shadow-[0_0_8px_#00f0ff]" />
          <h1 className="text-[#00f0ff] font-mono text-sm uppercase tracking-widest">
            ARIA.Core.Terminal
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-white/40 text-xs font-mono uppercase">
            Uplink:
          </span>
          <span
            className={`text-xs font-mono uppercase ${
              connectionState === 'connected'
                ? 'text-[#00f0ff]'
                : connectionState === 'streaming'
                ? 'text-[#0057ff] animate-pulse'
                : connectionState === 'reconnecting'
                ? 'text-[#ff9900] animate-pulse'
                : 'text-[#ff003c]'
            }`}
          >
            {connectionState}
          </span>
        </div>
      </div>

      {/* Feed Area */}
      <div
        ref={feedRef}
        className="flex-1 overflow-y-auto p-6 font-mono text-sm text-[#00f0ff] space-y-4 scrollbar-thin scrollbar-thumb-[#00f0ff]/20 scrollbar-track-transparent"
      >
        <AnimatePresence initial={false}>
          {messages.map((msg, idx) => (
            <motion.div
              key={msg.id || idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className={`flex flex-col ${
                msg.role === 'user' ? 'items-end' : 'items-start'
              }`}
            >
              <div
                className={`max-w-[80%] p-3 rounded-lg border ${
                  msg.role === 'user'
                    ? 'bg-[#0057ff]/10 border-[#0057ff]/30 text-white shadow-[0_0_15px_rgba(0,87,255,0.1)]'
                    : msg.role === 'system'
                    ? 'bg-[#ff003c]/10 border-[#ff003c]/30 text-[#ff003c] shadow-[0_0_15px_rgba(255,0,60,0.1)]'
                    : 'bg-[#00f0ff]/5 border-[#00f0ff]/20 text-[#00f0ff] shadow-[0_0_15px_rgba(0,240,255,0.1)]'
                }`}
              >
                <div className="text-[10px] uppercase tracking-widest opacity-50 mb-1">
                  {msg.role}
                </div>
                <div className="whitespace-pre-wrap leading-relaxed">
                  {msg.content}
                  {msg.role === 'assistant' &&
                    idx === messages.length - 1 &&
                    connectionState === 'streaming' && (
                      <motion.span
                        animate={{ opacity: [1, 0] }}
                        transition={{ repeat: Infinity, duration: 0.8 }}
                        className="inline-block w-2 h-4 bg-[#00f0ff] ml-1 align-middle"
                      />
                    )}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full opacity-30">
            Awaiting input...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex items-center gap-3 p-4 bg-slate-950/50 border-t border-[#00f0ff]/20">
        <div className="relative flex-1 flex items-center">
          <span className="absolute left-3 text-[#00f0ff]/50 font-mono text-sm">{'>'}</span>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Awaiting directive..."
            className="w-full bg-transparent border border-[#00f0ff]/30 rounded-lg py-3 pl-8 pr-12 text-[#00f0ff] font-mono text-sm focus:outline-none focus:border-[#00f0ff] focus:shadow-[0_0_15px_rgba(0,240,255,0.2)] transition-all placeholder:text-[#00f0ff]/30"
          />
          <button
            onClick={toggleListen}
            className={`absolute right-3 p-1.5 rounded-md transition-colors ${
              isListening
                ? 'text-[#ff003c] bg-[#ff003c]/10 animate-pulse'
                : 'text-[#00f0ff]/50 hover:text-[#00f0ff] hover:bg-[#00f0ff]/10'
            }`}
            title="Voice Input"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
              <path d="M8.25 4.5a3.75 3.75 0 117.5 0v8.25a3.75 3.75 0 11-7.5 0V4.5z" />
              <path d="M6 10.5a.75.75 0 01.75.75v1.5a5.25 5.25 0 1010.5 0v-1.5a.75.75 0 011.5 0v1.5a6.751 6.751 0 01-6 6.709v2.291h3a.75.75 0 010 1.5h-7.5a.75.75 0 010-1.5h3v-2.291a6.751 6.751 0 01-6-6.709v-1.5A.75.75 0 016 10.5z" />
            </svg>
          </button>
        </div>
        <button
          onClick={handleSend}
          disabled={!inputValue.trim()}
          className="px-6 py-3 bg-[#00f0ff]/10 border border-[#00f0ff]/30 rounded-lg text-[#00f0ff] font-mono text-sm uppercase tracking-widest hover:bg-[#00f0ff]/20 hover:shadow-[0_0_15px_rgba(0,240,255,0.3)] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </div>
    </div>
  );
};
