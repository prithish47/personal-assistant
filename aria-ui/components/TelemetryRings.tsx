'use client';

import { motion } from 'framer-motion';
import { useAriaStore } from '../store/useAriaStore';

const MetricRing = ({
  value,
  label,
  isWarning,
  unit = '%',
}: {
  value: number;
  label: string;
  isWarning: boolean;
  unit?: string;
}) => {
  const strokeColor = isWarning ? '#ff003c' : '#00f0ff';
  
  return (
    <div className="relative flex flex-col items-center justify-center w-32 h-32">
      <motion.svg 
        className={`absolute inset-0 w-full h-full opacity-70 ${
          isWarning 
            ? 'drop-shadow-[0_0_8px_rgba(255,0,60,0.8)] animate-pulse' 
            : 'drop-shadow-[0_0_8px_rgba(0,240,255,0.8)]'
        }`}
        viewBox="0 0 100 100"
        initial={{ rotate: 0 }}
        animate={{ rotate: 360 }}
        transition={{
          rotate: { repeat: Infinity, duration: 10, ease: 'linear' }
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
          animate={{ stroke: strokeColor }}
          transition={{ stroke: { duration: 0.5 } }}
        />
        <circle
          cx="50"
          cy="50"
          r="40"
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="1"
        />
      </motion.svg>
      <div className="z-10 flex flex-col items-center">
        <motion.span
          animate={{ color: strokeColor }}
          className="text-2xl font-bold tracking-widest font-mono"
        >
          {Math.round(value)}
          <span className="text-sm">{unit}</span>
        </motion.span>
        <span className="text-[10px] uppercase tracking-widest text-white/50 font-mono mt-1">
          {label}
        </span>
      </div>
    </div>
  );
};

export const TelemetryRings = () => {
  const { systemStats } = useAriaStore();

  return (
    <div className="flex flex-col gap-8 p-6 backdrop-blur-md bg-slate-950/80 border border-[#0057ff]/30 rounded-xl shadow-[0_0_20px_rgba(0,87,255,0.15)]">
      <h2 className="text-[#00f0ff] font-mono text-xs uppercase tracking-[0.2em] mb-2 border-b border-[#00f0ff]/20 pb-2">
        Hardware Telemetry
      </h2>
      <div className="flex flex-row lg:flex-col justify-between lg:justify-start gap-4 lg:gap-6 overflow-x-auto scrollbar-none">
        <MetricRing
          value={systemStats.cpu_temp}
          label="CPU Temp"
          unit="°C"
          isWarning={systemStats.cpu_temp > 90}
        />
        <MetricRing
          value={systemStats.gpu_temp}
          label="GPU Temp"
          unit="°C"
          isWarning={systemStats.gpu_temp > 85}
        />
        <MetricRing
          value={systemStats.vram_usage}
          label="VRAM Usage"
          unit="%"
          isWarning={systemStats.vram_usage > 95}
        />
      </div>
    </div>
  );
};
