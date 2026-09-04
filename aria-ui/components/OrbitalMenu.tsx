'use client';

import { motion } from 'framer-motion';
import { useAriaWebSocket } from '../hooks/useAriaWebSocket';

const CommandButton = ({
  label,
  intent,
  target,
  icon,
}: {
  label: string;
  intent: string;
  target?: string;
  icon?: string;
}) => {
  const { sendCommand } = useAriaWebSocket();

  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => sendCommand(intent, target)}
      className="relative shrink-0 flex items-center justify-between px-6 py-4 bg-black/40 border border-[#00f0ff]/30 rounded-lg group overflow-hidden transition-all duration-300 hover:shadow-[0_0_20px_rgba(0,240,255,0.4)] hover:border-[#00f0ff]"
    >
      <div className="absolute inset-0 bg-gradient-to-r from-[#00f0ff]/0 via-[#00f0ff]/10 to-[#00f0ff]/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      <div className="relative z-10 flex items-center gap-3">
        {icon && <span className="text-xl text-[#00f0ff]">{icon}</span>}
        <span className="text-sm whitespace-nowrap font-mono tracking-widest text-white uppercase group-hover:text-[#00f0ff] transition-colors">
          {label}
        </span>
      </div>
      <div className="w-1.5 h-1.5 rounded-full bg-[#00f0ff]/50 group-hover:bg-[#00f0ff] group-hover:animate-ping ml-4" />
    </motion.button>
  );
};

export const OrbitalMenu = () => {
  return (
    <div className="flex flex-col gap-4 p-6 backdrop-blur-md bg-black/40 border border-[#0057ff]/30 rounded-xl shadow-[0_0_20px_rgba(0,87,255,0.15)]">
      <h2 className="text-[#00f0ff] font-mono text-xs uppercase tracking-[0.2em] mb-2 border-b border-[#00f0ff]/20 pb-2">
        Quick Actions
      </h2>
      <div className="flex flex-row lg:flex-col gap-3 overflow-x-auto scrollbar-none pb-2 lg:pb-0">
        <CommandButton
          label="Launch Valorant"
          intent="OS_EXECUTE"
          target="Valorant"
          icon="🎮"
        />
        <CommandButton
          label="Open VS Code"
          intent="OS_EXECUTE"
          target="Code"
          icon="💻"
        />
        <CommandButton
          label="Clear Memory"
          intent="SYSTEM_CLEAR_MEMORY"
          icon="🧠"
        />
      </div>
    </div>
  );
};
