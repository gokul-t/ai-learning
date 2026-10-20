import ollama
import json


def run_chat_cli(model_name="qwen3.5:0.8b"):
    messages = [
        {
            "role": "system",
            "content": """You are a Java exception analyzer.

Analyze the exception provided by the user.

Return ONLY valid JSON with exactly these fields:

{
  "error_type": "...",
  "root_cause": "...",
  "suggested_fix": "...",
  "confidence": 0.0
}

confidence must be a number between 0 and 1.
Do not include markdown.
Do not include explanations outside the JSON.""",
        }
    ]
    print(f"Using model: {model_name}")
    user_input = input("\nYou: ").strip()
    messages.append({"role": "user", "content": user_input})
    response = ollama.chat(model=model_name, messages=messages, format="json")

    answer = response["message"]["content"]

    data = json.loads(answer)

    print("Error:", data["error_type"])
    print("Root cause:", data["root_cause"])
    print("Fix:", data["suggested_fix"])
    print("Confidence:", data["confidence"])


if __name__ == "__main__":
    run_chat_cli()
