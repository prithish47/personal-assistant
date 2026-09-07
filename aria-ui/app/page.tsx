'use client';

import { TerminalFeed } from '../components/TerminalFeed';
import { AgentStatusPanel } from '../components/AgentStatusPanel';
import { OrbitalMenu } from '../components/OrbitalMenu';
import { useAriaWebSocket } from '../hooks/useAriaWebSocket';

export default function DashboardPage() {
  // Initialize WebSocket connection at the page level
  useAriaWebSocket();

  return (
    <main className="relative min-h-screen bg-black overflow-hidden selection:bg-[#00f0ff]/30 selection:text-[#00f0ff]">
      {/* CRT Scanline Overlay */}
      <div className="pointer-events-none fixed inset-0 z-50 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_4px,3px_100%] opacity-20" />
      
      {/* Radial Gradient Background */}
      <div className="pointer-events-none fixed inset-0 z-0 bg-[radial-gradient(circle_at_center,rgba(0,87,255,0.05)_0%,rgba(0,0,0,1)_80%)]" />

      {/* Main Grid Layout */}
      <div className="relative z-10 w-full h-[100dvh] max-w-[1920px] mx-auto p-4 md:p-6 lg:p-10 flex flex-col lg:grid lg:grid-cols-12 gap-4 lg:gap-8 overflow-hidden">
        
        {/* Top/Left Column: Telemetry & Actions */}
        <div className="flex-none lg:col-span-3 flex flex-col gap-4 lg:gap-8 h-auto lg:h-full z-10">
          <div className="flex items-center gap-4 mb-0 lg:mb-4">
            <div className="w-10 h-10 lg:w-12 lg:h-12 rounded-full border-2 border-[#00f0ff] flex items-center justify-center shadow-[0_0_15px_rgba(0,240,255,0.4)]">
              <div className="w-6 h-6 lg:w-8 lg:h-8 rounded-full bg-[#00f0ff]/20 animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl lg:text-2xl font-bold text-white font-mono tracking-widest drop-shadow-[0_0_8px_rgba(0,240,255,0.8)]">
                ARIA
              </h1>
              <p className="text-[8px] lg:text-[10px] text-[#00f0ff] font-mono uppercase tracking-[0.3em]">
                System Active
              </p>
            </div>
          </div>

          <div className="flex flex-col lg:flex-1 gap-4 lg:gap-8 overflow-y-auto overflow-x-hidden scrollbar-none pb-2 lg:pb-0">
            <AgentStatusPanel />
            <OrbitalMenu />
          </div>
        </div>

        {/* Bottom/Right Column: Terminal Feed */}
        <div className="flex-1 lg:col-span-9 h-full min-h-0">
          <TerminalFeed />
        </div>

      </div>
    </main>
  );
}
