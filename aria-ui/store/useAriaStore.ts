import { create } from 'zustand';

export type MessageRole = 'user' | 'assistant' | 'system';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
}

export interface SystemStats {
  cpu_temp: number;
  gpu_temp: number;
  vram_usage: number; // Percentage
}

export type ConnectionState = 'connected' | 'disconnected' | 'streaming' | 'reconnecting';

interface AriaState {
  messages: Message[];
  systemStats: SystemStats;
  connectionState: ConnectionState;

  // Actions
  addMessage: (msg: Message) => void;
  appendTokenToLastMessage: (token: string) => void;
  updateSystemStats: (stats: Partial<SystemStats>) => void;
  setConnectionState: (state: ConnectionState) => void;
  clearMessages: () => void;
}

export const useAriaStore = create<AriaState>((set) => ({
  messages: [],
  systemStats: {
    cpu_temp: 45,
    gpu_temp: 50,
    vram_usage: 20,
  },
  connectionState: 'disconnected',

  addMessage: (msg) =>
    set((state) => ({ messages: [...state.messages, msg] })),

  appendTokenToLastMessage: (token) =>
    set((state) => {
      const msgs = [...state.messages];
      if (msgs.length > 0) {
        const lastMsg = msgs[msgs.length - 1];
        if (lastMsg.role === 'assistant') {
          lastMsg.content += token;
        }
      }
      return { messages: msgs };
    }),

  updateSystemStats: (stats) =>
    set((state) => ({
      systemStats: { ...state.systemStats, ...stats },
    })),

  setConnectionState: (connectionState) => set({ connectionState }),

  clearMessages: () => set({ messages: [] }),
}));
