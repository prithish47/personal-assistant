# voice/listener.py
import speech_recognition as sr

recognizer = sr.Recognizer()

def listen() -> str:
    """Listen for voice input with proper timeout handling."""
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("[ARIA] Listening... (speak now)")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        
        text = recognizer.recognize_google(audio)
        print(f"[ARIA] Heard: {text}")
        return text

    except sr.WaitTimeoutError:
        print("[ARIA] No speech detected.")
        return ""
    except sr.UnknownValueError:
        print("[ARIA] Couldn't understand that sir.")
        return ""
    except sr.RequestError as e:
        print(f"[ARIA] Google Speech error: {e}")
        return ""
    except Exception as e:
        print(f"[ARIA] Listener error: {e}")
        return ""