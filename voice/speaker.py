# voice/speaker.py
import pyttsx3
import threading

engine = pyttsx3.init()

# tune the voice — more Jarvis-like
engine.setProperty('rate', 185)      # speed
engine.setProperty('volume', 1.0)    # max volume

# pick a male voice if available
voices = engine.getProperty('voices')
for voice in voices:
    if 'male' in voice.name.lower() or 'david' in voice.name.lower() or 'mark' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

def speak(text: str):
    """Speak text out loud. Runs in background so terminal stays responsive."""
    def _speak():
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=_speak)
    thread.start()
    thread.join()

def list_voices():
    """Debug helper — see all available voices."""
    for v in voices:
        print(f"ID: {v.id} | Name: {v.name}")