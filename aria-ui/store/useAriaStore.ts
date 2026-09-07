import { create } from 'zustand';

export type MessageRole = 'user' | 'assistant' | 'system' | 'tool' | 'error';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
}

/** Real, backend-sourced agent status — replaces the old fabricated hardware telemetry. */
export interface AgentStatus {
  activeAgent: string | null;
  lastRoutingReason: string | null;
  lastConfidence: number | null;
  toolCallCount: number;
}

export type ConnectionState = 'connecting' | 'connected' | 'streaming' | 'reconnecting' | 'auth_error' | 'disconnected';

/**
 * The voice control's own state machine (STEP 5). 'transcribing', 'speaking', and 'idle'
 * mirror the server's `VoiceState` (see lib/protocol.ts) one-to-one; 'listening' (recording
 * in progress) and 'thinking' (transcript received, orchestrator working) are client-only --
 * the server has no concept of either, since a voice turn only exists to it as a single
 * voice_input message. 'error' covers both a voice_error event and a local mic failure.
 */
export type VoiceUiState = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'speaking' | 'error';

interface AriaState {
  messages: Message[];
  agentStatus: AgentStatus;
  connectionState: ConnectionState;
  voiceState: VoiceUiState;
  voiceError: string | null;

  // Actions
  addMessage: (msg: Message) => void;
  appendToken: (messageId: string, token: string) => void;
  setAgentStatus: (status: Partial<AgentStatus>) => void;
  incrementToolCallCount: () => void;
  setConnectionState: (state: ConnectionState) => void;
  setVoiceState: (state: VoiceUiState) => void;
  setVoiceError: (message: string | null) => void;
  clearMessages: () => void;
}

export const useAriaStore = create<AriaState>((set) => ({
  messages: [],
  agentStatus: {
    activeAgent: null,
    lastRoutingReason: null,
    lastConfidence: null,
    toolCallCount: 0,
  },
  connectionState: 'disconnected',
  voiceState: 'idle',
  voiceError: null,

  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),

  // Targets a specific message by id, set at send-time, rather than
  // assuming "the last message" — routing/tool events can be appended
  // to the feed after the assistant placeholder but before the first token.
  appendToken: (messageId, token) =>
    set((state) => ({
      messages: state.messages.map((message) =>
        message.id === messageId ? { ...message, content: message.content + token } : message
      ),
    })),

  setAgentStatus: (status) =>
    set((state) => ({
      agentStatus: { ...state.agentStatus, ...status },
    })),

  incrementToolCallCount: () =>
    set((state) => ({
      agentStatus: { ...state.agentStatus, toolCallCount: state.agentStatus.toolCallCount + 1 },
    })),

  setConnectionState: (connectionState) => set({ connectionState }),

  setVoiceState: (voiceState) => set({ voiceState }),

  setVoiceError: (voiceError) => set({ voiceError }),

  clearMessages: () => set({ messages: [] }),
}));
