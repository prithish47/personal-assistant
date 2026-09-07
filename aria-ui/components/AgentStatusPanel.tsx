'use client';

import { motion } from 'framer-motion';
import { useAriaStore } from '../store/useAriaStore';

/**
 * Replaces the old fabricated CPU/GPU/VRAM rings (there is no hardware
 * telemetry event from the backend) with the router's actual last
 * confidence score — same ring visual, real backend-sourced data.
 */
const ConfidenceRing = ({ value }: { value: number | null }) => {
  const strokeColor = '#00f0ff';
  const displayValue = value === null ? '—' : `${Math.round(value * 100)}%`;

  return (
    <div className="relative flex flex-col items-center justify-center w-32 h-32">
      <motion.svg
        className="absolute inset-0 w-full h-full opacity-70 drop-shadow-[0_0_8px_rgba(0,240,255,0.8)]"
        viewBox="0 0 100 100"
        initial={{ rotate: 0 }}
        animate={{ rotate: 360 }}
        transition={{
          rotate: { repeat: Infinity, duration: 10, ease: 'linear' },
        }}
        style={{ originX: 0.5, originY: 0.5 }}
      >
        <motion.circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke={strokeColor}
          strokeWidth="2"
          strokeDasharray="10 5"
        />
        <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
      </motion.svg>
      <div className="z-10 flex flex-col items-center">
        <span className="text-2xl font-bold tracking-widest font-mono" style={{ color: strokeColor }}>
          {displayValue}
        </span>
        <span className="text-[10px] uppercase tracking-widest text-white/50 font-mono mt-1">
          Routing Confidence
        </span>
      </div>
    </div>
  );
};

const StatusRow = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-center justify-between px-1">
    <span className="text-[10px] uppercase tracking-widest text-white/40 font-mono">{label}</span>
    <span className="text-sm font-mono text-[#00f0ff] truncate max-w-[60%] text-right" title={value}>
      {value}
    </span>
  </div>
);

export const AgentStatusPanel = () => {
  const { agentStatus } = useAriaStore();

  return (
    <div className="flex flex-col gap-6 p-6 backdrop-blur-md bg-slate-950/80 border border-[#0057ff]/30 rounded-xl shadow-[0_0_20px_rgba(0,87,255,0.15)]">
      <h2 className="text-[#00f0ff] font-mono text-xs uppercase tracking-[0.2em] mb-2 border-b border-[#00f0ff]/20 pb-2">
        Agent Status
      </h2>
      <div className="flex justify-center">
        <ConfidenceRing value={agentStatus.lastConfidence} />
      </div>
      <div className="flex flex-col gap-2">
        <StatusRow label="Active Agent" value={agentStatus.activeAgent ?? 'None'} />
        <StatusRow label="Tool Calls" value={String(agentStatus.toolCallCount)} />
        <StatusRow label="Last Reason" value={agentStatus.lastRoutingReason ?? '—'} />
      </div>
    </div>
  );
};
