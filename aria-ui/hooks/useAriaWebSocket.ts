import { useEffect, useRef, useCallback } from 'react';
import { useAriaStore } from '../store/useAriaStore';

export const useAriaWebSocket = (url: string = 'ws://localhost:8000/ws/aria') => {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const tokenBuffer = useRef('');
  const flushTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const {
    appendTokenToLastMessage,
    updateSystemStats,
    setConnectionState,
    addMessage,
  } = useAriaStore();

  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnectionState('connected');
      reconnectAttempt.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'token':
            setConnectionState('streaming');
            tokenBuffer.current += data.content;
            if (!flushTimeoutRef.current) {
              flushTimeoutRef.current = setTimeout(() => {
                appendTokenToLastMessage(tokenBuffer.current);
                tokenBuffer.current = '';
                flushTimeoutRef.current = null;
              }, 50);
            }
            break;
          case 'message_end':
            if (flushTimeoutRef.current) {
              clearTimeout(flushTimeoutRef.current);
              flushTimeoutRef.current = null;
            }
            if (tokenBuffer.current) {
              appendTokenToLastMessage(tokenBuffer.current);
              tokenBuffer.current = '';
            }
            setConnectionState('connected');
            break;
          case 'telemetry':
            updateSystemStats({
              cpu_temp: data.data.cpu_temp,
              gpu_temp: data.data.gpu_temp,
              vram_usage: data.data.vram_usage,
            });
            break;
          case 'message':
            addMessage({
              id: crypto.randomUUID(),
              role: data.role,
              content: data.content,
              timestamp: Date.now(),
            });
            break;
          default:
            console.warn('Unknown message type:', data.type);
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message', err);
      }
    };

    ws.onclose = () => {
      setConnectionState('reconnecting');
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempt.current), 30000);
      reconnectAttempt.current += 1;
      reconnectTimeoutRef.current = setTimeout(connect, delay);
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
      ws.close();
    };

    socketRef.current = ws;
  }, [url, appendTokenToLastMessage, updateSystemStats, setConnectionState, addMessage]);

  useEffect(() => {
    connect();
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (flushTimeoutRef.current) {
        clearTimeout(flushTimeoutRef.current);
      }
    };
  }, [connect]);

  const sendCommand = (intent: string, target?: string) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      const payload: Record<string, string> = { intent };
      if (target) payload.target = target;
      socketRef.current.send(JSON.stringify(payload));
    } else {
      console.warn('Cannot send command, WebSocket is not open.');
    }
  };

  const sendMessage = (content: string) => {
    addMessage({
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now(),
    });
    addMessage({
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    });
    
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'user_input', content }));
    }
  };

  return { sendCommand, sendMessage };
};
