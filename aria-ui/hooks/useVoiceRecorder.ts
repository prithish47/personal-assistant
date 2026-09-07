'use client';

import { useCallback, useRef, useState } from 'react';

/**
 * Wraps browser microphone capture (MediaRecorder) into simple start/stop calls.
 *
 * Deliberately knows nothing about the WebSocket wire format -- it only ever
 * hands back one complete recorded Blob per `stop()` call, matching the
 * backend's one-message-per-utterance `voice_input` contract (see
 * aria/app.py's `handle_voice_input`): the whole recording is buffered in
 * the browser and sent once, not streamed frame by frame.
 */
export const useVoiceRecorder = () => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Microphone access was denied.');
      setIsRecording(false);
    }
  }, []);

  /** Stops recording and resolves with the complete audio, or null if nothing was captured. */
  const stop = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state === 'inactive') {
        resolve(null);
        return;
      }
      recorder.onstop = () => {
        const blob = chunksRef.current.length > 0 ? new Blob(chunksRef.current, { type: recorder.mimeType }) : null;
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        mediaRecorderRef.current = null;
        setIsRecording(false);
        resolve(blob);
      };
      recorder.stop();
    });
  }, []);

  return { isRecording, error, start, stop };
};
