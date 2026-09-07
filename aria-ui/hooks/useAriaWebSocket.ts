'use client';

import { useCallback, useEffect, useRef } from 'react';
import { useAriaStore } from '../store/useAriaStore';
import type { IncomingEvent, OutgoingChatMessage, OutgoingVoiceMessage } from '../lib/protocol';

const WS_URL = process.env.NEXT_PUBLIC_ARIA_WS_URL ?? 'ws://127.0.0.1:8000/v1/ws';
const API_KEY = process.env.NEXT_PUBLIC_ARIA_API_KEY ?? '';

/** WebSocket close code the backend uses for both bad credentials and rate limiting (see aria/api/security.py). */
const POLICY_REJECTION_CLOSE_CODE = 1008;

/** One recorded utterance's audio, base64-encoded for the JSON wire format (see aria/app.py). */
const blobToBase64 = (blob: Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string; // "data:<mime>;base64,<data>"
      resolve(result.slice(result.indexOf(',') + 1));
    };
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read recorded audio'));
    reader.readAsDataURL(blob);
  });

/** The inverse of the server's `base64.b64encode(audio).decode("ascii")` for a synthesized chunk. */
const base64ToBlob = (base64: string, mimeType: string): Blob => {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mimeType });
};

/**
 * Owns the `/v1/ws` connection: this is the only place that knows the wire
 * format. Components never see raw events — they read state from the store
 * and call `sendMessage`.
 */
export const useAriaWebSocket = () => {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const conversationId = useRef(crypto.randomUUID());
  const pendingAssistantId = useRef<string | null>(null);

  // Queued object URLs for synthesized speech, played strictly one at a time so sentence
  // chunks never overlap; `audioElementRef` is a single reused <audio>, not part of the DOM
  // tree -- playing audio doesn't require rendering anything.
  const audioQueueRef = useRef<string[]>([]);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const isPlayingRef = useRef(false);

  const {
    addMessage,
    appendToken,
    setAgentStatus,
    incrementToolCallCount,
    setConnectionState,
    setVoiceState,
    setVoiceError,
  } = useAriaStore();

  const playNextChunk = useCallback(() => {
    if (isPlayingRef.current) return;
    const nextUrl = audioQueueRef.current.shift();
    if (!nextUrl) return;
    isPlayingRef.current = true;
    const audioEl = audioElementRef.current ?? new Audio();
    audioElementRef.current = audioEl;
    audioEl.src = nextUrl;
    audioEl.onended = () => {
      URL.revokeObjectURL(nextUrl);
      isPlayingRef.current = false;
      playNextChunk();
    };
    audioEl.play().catch(() => {
      // Autoplay can be blocked until the user has interacted with the page at all; since
      // clicking the mic button is itself a user gesture, this should be rare in practice.
      URL.revokeObjectURL(nextUrl);
      isPlayingRef.current = false;
    });
  }, []);

  /** Stops playback immediately and drops anything still queued -- used to interrupt speech. */
  const stopSpeaking = useCallback(() => {
    audioQueueRef.current.forEach((url) => URL.revokeObjectURL(url));
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    audioElementRef.current?.pause();
    setVoiceState('idle');
  }, [setVoiceState]);

  const handleEvent = useCallback(
    (event: IncomingEvent) => {
      switch (event.type) {
        case 'routing_decision':
          setAgentStatus({
            activeAgent: event.data.agent,
            lastRoutingReason: event.data.reason,
            lastConfidence: event.data.confidence,
          });
          addMessage({
            id: crypto.randomUUID(),
            role: 'system',
            content: `Routed to ${event.data.agent} (${Math.round(event.data.confidence * 100)}%) — ${event.data.reason}`,
            timestamp: Date.now(),
          });
          break;

        case 'agent_started':
          setAgentStatus({ activeAgent: event.data.agent });
          break;

        case 'tool_call':
          addMessage({
            id: crypto.randomUUID(),
            role: 'tool',
            content: `→ ${event.data.name}(${JSON.stringify(event.data.arguments)})`,
            timestamp: Date.now(),
          });
          break;

        case 'tool_result':
          incrementToolCallCount();
          addMessage({
            id: crypto.randomUUID(),
            role: 'tool',
            content:
              event.data.status === 'succeeded'
                ? `✓ ${event.data.tool_name} → ${JSON.stringify(event.data.result)}`
                : `✗ ${event.data.tool_name} ${event.data.status}: ${event.data.error ?? 'no result'}`,
            timestamp: Date.now(),
          });
          break;

        case 'token':
          setConnectionState('streaming');
          if (pendingAssistantId.current) {
            appendToken(pendingAssistantId.current, event.content);
          }
          break;

        case 'complete':
          pendingAssistantId.current = null;
          setConnectionState('connected');
          // Safety net: a turn that completed with no text at all (so `speak()` was never
          // called server-side, see aria/app.py) would otherwise leave the voice control
          // stuck on "thinking" forever.
          if (useAriaStore.getState().voiceState === 'thinking') setVoiceState('idle');
          break;

        case 'error':
          pendingAssistantId.current = null;
          addMessage({
            id: crypto.randomUUID(),
            role: 'error',
            content: `${event.code}: ${event.message}`,
            timestamp: Date.now(),
          });
          setConnectionState('connected');
          if (useAriaStore.getState().voiceState !== 'idle') setVoiceState('idle');
          break;

        case 'voice_status':
          setVoiceState(event.data.state);
          break;

        case 'transcription': {
          // Mirrors sendMessage() below: a voice turn creates its own user + assistant
          // placeholder messages here, since (unlike typed input) the client never echoed
          // this text itself -- it only learns it once the server has transcribed it.
          const assistantId = crypto.randomUUID();
          pendingAssistantId.current = assistantId;
          addMessage({ id: crypto.randomUUID(), role: 'user', content: event.data.text, timestamp: Date.now() });
          addMessage({ id: assistantId, role: 'assistant', content: '', timestamp: Date.now() });
          setVoiceState('thinking');
          break;
        }

        case 'audio_chunk': {
          const blob = base64ToBlob(event.data.audio_base64, event.data.mime_type);
          audioQueueRef.current.push(URL.createObjectURL(blob));
          playNextChunk();
          break;
        }

        case 'voice_error':
          setVoiceState('error');
          setVoiceError(event.data.message);
          addMessage({
            id: crypto.randomUUID(),
            role: 'error',
            content: `Voice: ${event.data.message}`,
            timestamp: Date.now(),
          });
          break;

        default:
          console.warn('Unknown event type:', event);
      }
    },
    [
      addMessage,
      appendToken,
      setAgentStatus,
      incrementToolCallCount,
      setConnectionState,
      setVoiceState,
      setVoiceError,
      playNextChunk,
    ]
  );

  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return;

    setConnectionState('connecting');
    const ws = new WebSocket(`${WS_URL}?token=${encodeURIComponent(API_KEY)}`);

    ws.onopen = () => {
      setConnectionState('connected');
      reconnectAttempt.current = 0;
    };

    ws.onmessage = (message) => {
      try {
        handleEvent(JSON.parse(message.data) as IncomingEvent);
      } catch (err) {
        console.error('Failed to parse WebSocket message', err);
      }
    };

    ws.onclose = (closeEvent) => {
      if (closeEvent.code === POLICY_REJECTION_CLOSE_CODE) {
        // Invalid API key or rate-limited: retrying immediately would just repeat the rejection.
        setConnectionState('auth_error');
        return;
      }
      setConnectionState('reconnecting');
      const delay = Math.min(1000 * 2 ** reconnectAttempt.current, 30000);
      reconnectAttempt.current += 1;
      reconnectTimeoutRef.current = setTimeout(connect, delay);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      ws.close();
    };

    socketRef.current = ws;
  }, [handleEvent, setConnectionState]);

  useEffect(() => {
    connect();
    return () => {
      socketRef.current?.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [connect]);

  const sendMessage = useCallback(
    (content: string) => {
      const trimmed = content.trim();
      if (!trimmed) return;

      addMessage({ id: crypto.randomUUID(), role: 'user', content: trimmed, timestamp: Date.now() });

      if (socketRef.current?.readyState !== WebSocket.OPEN) {
        addMessage({
          id: crypto.randomUUID(),
          role: 'error',
          content: 'Not connected — message was not sent.',
          timestamp: Date.now(),
        });
        return;
      }

      const assistantId = crypto.randomUUID();
      pendingAssistantId.current = assistantId;
      addMessage({ id: assistantId, role: 'assistant', content: '', timestamp: Date.now() });

      const payload: OutgoingChatMessage = {
        content: trimmed,
        conversation_id: conversationId.current,
        approval_token: null,
      };
      socketRef.current.send(JSON.stringify(payload));
    },
    [addMessage]
  );

  /**
   * Sends one complete recorded utterance. Unlike `sendMessage`, this does not add a user
   * message itself -- the client doesn't know the words yet, only the audio -- the 'user'
   * message is created from the server's `transcription` event instead (see handleEvent above).
   */
  const sendVoiceInput = useCallback(
    async (audioBlob: Blob) => {
      if (socketRef.current?.readyState !== WebSocket.OPEN) {
        addMessage({
          id: crypto.randomUUID(),
          role: 'error',
          content: 'Not connected — recording was not sent.',
          timestamp: Date.now(),
        });
        setVoiceState('idle');
        return;
      }

      const audioBase64 = await blobToBase64(audioBlob);
      const payload: OutgoingVoiceMessage = {
        type: 'voice_input',
        conversation_id: conversationId.current,
        approval_token: null,
        audio_base64: audioBase64,
        mime_type: audioBlob.type || 'audio/webm',
      };
      socketRef.current.send(JSON.stringify(payload));
    },
    [addMessage, setVoiceState]
  );

  return { sendMessage, sendVoiceInput, stopSpeaking, reconnect: connect };
};
