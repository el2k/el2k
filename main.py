from agent import AgentRuntime

def main():
    runtime = AgentRuntime()

    print("=" * 60)
    print("  Minimal Agent Framework - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  'new'     - Create a new session")
    print("  'list'    - List all sessions")
    print("  'quit'    - Exit")
    print("=" * 60)

    current_session = None

    while True:
        try:
            user_input = input("\n[Agent] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "new":
            current_session = runtime.start_session()
            print(f"Created new session: {current_session}")
            continue

        if user_input.lower() == "list":
            sessions = runtime.list_sessions()
            print(f"Active sessions ({len(sessions)}):")
            for sid in sessions:
                marker = " <-- current" if sid == current_session else ""
                print(f"  {sid}{marker}")
            continue

        if current_session is None:
            print("Create a session first with 'new'")
            continue

        result = runtime.chat(current_session, user_input)
        print(f"\n[Response] {result['response']}")
        if result.get("tool_calls"):
            print(f"[Tools used] {result['tool_calls']}")
        print(f"[Turn] {result['turn']}")

if __name__ == "__main__":
    main()