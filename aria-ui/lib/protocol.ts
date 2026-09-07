/**
 * Wire types for the ARIA WebSocket contract (`/v1/ws`).
 *
 * These mirror `aria/orchestration/orchestrator.py` and
 * `aria/domain/models.py` exactly. If the backend event shapes change,
 * update this file first — every consumer below is typed against it.
 */

/** What the client sends for every typed chat turn. Unchanged by voice. */
export interface OutgoingChatMessage {
  content: string;
  conversation_id: string;
  approval_token: string | null;
}

/**
 * What the client sends for one complete recorded voice utterance.
 *
 * One message per recording (start-to-stop), not a stream of small audio
 * frames -- mirrors `aria/app.py`'s `handle_voice_input`, which expects one
 * full payload it can decode and transcribe in a single call.
 */
export interface OutgoingVoiceMessage {
  type: 'voice_input';
  conversation_id: string;
  approval_token: string | null;
  audio_base64: string;
  mime_type: string;
}

/** Mirrors `aria.domain.models.ToolStatus`. */
export type ToolStatus = 'pending' | 'running' | 'succeeded' | 'denied' | 'failed';

/** Mirrors `aria.domain.models.ToolExecutionRecord.model_dump(mode="json")`. */
export interface ToolExecutionRecord {
  id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  status: ToolStatus;
  result: Record<string, unknown> | null;
  error: string | null;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
}

export interface RoutingDecisionEvent {
  type: 'routing_decision';
  data: { agent: string; reason: string; confidence: number };
}

export interface AgentStartedEvent {
  type: 'agent_started';
  data: { agent: string };
}

export interface ToolCallEvent {
  type: 'tool_call';
  data: { name: string; arguments: Record<string, unknown> };
}

export interface ToolResultEvent {
  type: 'tool_result';
  data: ToolExecutionRecord;
}

export interface TokenEvent {
  type: 'token';
  content: string;
}

export interface CompleteEvent {
  type: 'complete';
  data: { agent: string; conversation_id: string };
}

export interface ErrorEvent {
  type: 'error';
  code: string;
  message: string;
}

/** Server-reported voice pipeline phase; a strict subset of the UI's own `VoiceUiState`
 * (see useAriaStore.ts) -- the server never knows about the client-only 'listening',
 * 'thinking', or 'error' phases, which only exist before/between voice_input messages. */
export type VoiceState = 'transcribing' | 'speaking' | 'idle';

export interface VoiceStatusEvent {
  type: 'voice_status';
  data: { state: VoiceState };
}

/** The transcribed text of one voice_input, re-entering the pipeline exactly like typed input. */
export interface TranscriptionEvent {
  type: 'transcription';
  data: { text: string };
}

/** One synthesized speech chunk (see aria/voice/tts.py's sentence-aligned chunking). */
export interface AudioChunkEvent {
  type: 'audio_chunk';
  data: { audio_base64: string; mime_type: string; index: number; final: boolean };
}

/** A voice-pipeline-specific failure (STT/TTS/payload validation) -- never the chat turn itself. */
export interface VoiceErrorEvent {
  type: 'voice_error';
  data: { code: string; message: string };
}

/** Every event JanusOrchestrator (and the pre-orchestrator validation check) can send. */
export type IncomingEvent =
  | RoutingDecisionEvent
  | AgentStartedEvent
  | ToolCallEvent
  | ToolResultEvent
  | TokenEvent
  | CompleteEvent
  | ErrorEvent
  | VoiceStatusEvent
  | TranscriptionEvent
  | AudioChunkEvent
  | VoiceErrorEvent;
