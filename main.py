# main.py
from orchestrator import route_stream
from config import ARIA_NAME, USER_NAME
from llm.ollama import unload_model
from voice.speaker import speak
from voice.listener import listen

SHUTDOWN_COMMANDS = ["shutdown()"]

def shutdown(voice_mode: bool = False):
    print(f"\n{ARIA_NAME}: Unloading from memory. Goodnight sir.\n")
    if voice_mode:
        speak("Unloading from memory. Goodnight sir.")
    unload_model()

def get_mode() -> bool:
    """Ask user to select mode on startup. Returns True if voice mode."""
    print(f"\n{'='*50}")
    print(f"  {ARIA_NAME} — Online")
    print(f"  Serving {USER_NAME}.")
    print(f"{'='*50}")
    print("\n  [1] Text Mode")
    print("  [2] Voice Mode")
    print()

    while True:
        choice = input("  Select mode (1 or 2): ").strip()
        if choice == "1":
            print(f"\n[ARIA] Text mode activated. Type away sir.\n")
            return False
        elif choice == "2":
            print(f"\n[ARIA] Voice mode activated. Speak when prompted sir.\n")
            speak(f"ARIA online. Voice mode activated. Good to see you {USER_NAME}.")
            return True
        else:
            print("  Invalid choice. Enter 1 or 2.")

def main():
    voice_mode = get_mode()
    conversation_history = []

    while True:
        try:
            if voice_mode:
                # voice input
                user_input = listen()
                if not user_input:
                    continue
            else:
                # text input
                print(f"{USER_NAME}: ", end="", flush=True)
                user_input = input().strip()
                if not user_input:
                    continue

            # shutdown check
            if user_input.lower() in SHUTDOWN_COMMANDS:
                shutdown(voice_mode)
                break

            # streaming output
            print(f"\n{ARIA_NAME}: ", end="", flush=True)
            full_response = ""

            for token in route_stream(user_input, conversation_history):
                print(token, end="", flush=True)
                full_response += token

            print()

            # speak only in voice mode
            if voice_mode:
                speak(full_response)

            # update history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            shutdown(voice_mode)
            break

if __name__ == "__main__":
    main()