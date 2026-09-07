"""Local speech I/O (STT/TTS) for the voice transport layer -- not a reasoning capability.

Nothing here talks to Qwen-VL or any other LLM. `stt.py` turns recorded audio
into plain text that re-enters the existing JANUS pipeline exactly like
typed input; `tts.py` turns a completed agent response back into audio.
"""

from __future__ import annotations
