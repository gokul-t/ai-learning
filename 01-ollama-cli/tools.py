import ollama
from pathlib import Path
import json


MODEL = "qwen3.5:0.8b"  # Change this to your Ollama model


# --------------------------------------------------
# Tools
# --------------------------------------------------


def list_files():
    """List files in the current directory."""
    return [path.name for path in Path(".").iterdir() if path.is_file()]


def get_current_directory():
    """Get the current working directory."""
    return str(Path.cwd())


def read_file(filename):
    """Read a text file from the current directory."""

    path = Path(filename)

    if not path.exists():
        return f"File '{filename}' does not exist."

    if not path.is_file():
        return f"'{filename}' is not a file."

    try:
        return path.read_text()
    except UnicodeDecodeError:
        return f"Cannot read '{filename}' because it is not a text file."


# --------------------------------------------------
# Tool definitions given to the LLM
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the current directory.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_directory",
            "description": "Get the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
]


# --------------------------------------------------
# Map tool names to actual Python functions
# --------------------------------------------------

available_tools = {
    "list_files": list_files,
    "get_current_directory": get_current_directory,
    "read_file": read_file,
}


# --------------------------------------------------
# Conversation
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": """
You are a helpful local developer assistant.

You have access to tools that allow you to inspect
the current project.

Use tools when they are necessary to answer the user's question.

Do not claim that you inspected a file or directory
unless you actually used the appropriate tool.
""",
    }
]


# --------------------------------------------------
# Main chat loop
# --------------------------------------------------

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Ask the model what to do
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

    # Add the assistant response to conversation
    messages.append(response["message"])

    # --------------------------------------------------
    # Check whether the model wants to call a tool
    # --------------------------------------------------

    tool_calls = response["message"].get("tool_calls", [])

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]

            arguments = tool_call["function"].get("arguments", {})

            print(f"\n[AI wants to call: {function_name}]")

            print(f"[Arguments: {arguments}]")

            # Find the Python function
            function = available_tools.get(function_name)

            if function is None:
                result = f"Unknown tool: {function_name}"

            else:
                try:
                    # Functions without arguments
                    if function_name in [
                        "list_files",
                        "get_current_directory",
                    ]:
                        result = function()

                    # Functions with arguments
                    elif function_name == "read_file":
                        filename = arguments["filename"]
                        result = function(filename)

                    else:
                        result = "Tool execution not implemented."

                except Exception as e:
                    result = f"Tool execution failed: {e}"

            print(f"[Tool result: {result}]")

            # Send tool result back to the model
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                }
            )

        # --------------------------------------------------
        # Ask LLM to generate final answer
        # --------------------------------------------------

        final_response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        messages.append(final_response["message"])

        print("\nAI:", final_response["message"]["content"])

    else:
        # No tool was needed
        print("\nAI:", response["message"]["content"])
