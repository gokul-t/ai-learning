import ollama


def run_chat_cli(model_name="qwen3.5:0.8b"):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print(f"Using model: {model_name}")
    print("Type '/exit' to quit the chat.")
    print("Type '/clear' to clear the chat history.")
    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        else:
            if user_input.lower() == "/exit":
                print("Exiting the chat. Goodbye!")
                break
            elif user_input.lower() == "/clear":
                messages = [messages[0]]  # Keep only the system message
                print("Chat history cleared.")
                continue
            else:
                messages.append({"role": "user", "content": user_input})
                stream = ollama.chat(model=model_name, messages=messages, stream=True)
                print("Assistant:", flush=True)
                assistant_response = ""
                thinking_printed = False
                response_printed = False

                for chunk in stream:
                    message = {}
                    try:
                        if isinstance(chunk, dict):
                            message = chunk.get("message", {}) or {}
                        else:
                            try:
                                message = chunk["message"] or {}
                            except Exception:
                                message = getattr(chunk, "message", {}) or {}
                    except Exception:
                        message = {}

                    if isinstance(message, dict):
                        thinking = message.get("thinking", "") or ""
                        content = message.get("content", "") or ""
                    else:
                        thinking = getattr(message, "thinking", "") or ""
                        content = getattr(message, "content", "") or ""

                    if thinking:
                        if not thinking_printed:
                            print("\nThinking:", end="", flush=True)
                            thinking_printed = True
                        print(thinking, end="", flush=True)

                    if content:
                        if not response_printed:
                            print("\nResponse:", end="", flush=True)
                            response_printed = True
                        print(content, end="", flush=True)
                        assistant_response += str(content)

                print()
                messages.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    run_chat_cli()
