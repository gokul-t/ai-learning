from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

# model used elsewhere in this project
model_name = "unsloth/SmolLM2-1.7B-Instruct"

# --- Sample dataset ---
# A small in-memory sample dataset suitable for instruction tuning / few-shot examples.
# Each example contains an 'input' (instruction or question) and an 'output' (desired response).
dataset = [
    {
        "id": "1",
        "input": "Summarize the following paragraph:\n\nNeovim is a modern fork of Vim focused on extensibility and usability. It preserves Vim's modal editing model while providing a better plugin architecture.",
        "output": "Neovim is a modern, extensible fork of Vim that keeps modal editing but improves plugin architecture and usability.",
    },
    {
        "id": "2",
        "input": "Translate the following sentence into French:\n\nThe weather is lovely today.",
        "output": "Il fait beau aujourd'hui.",
    },
    {
        "id": "3",
        "input": "Extract keywords from the sentence:\n\nDeploy the web app using Docker and Kubernetes for scalable production.",
        "output": "Keywords: deploy, web app, Docker, Kubernetes, scalable, production",
    },
    {
        "id": "4",
        "input": "Convert the following instruction to a concise checklist:\n\nTo set up a Python virtual environment: create a venv, activate it, install dependencies from requirements.txt, and run tests.",
        "output": "1. python -m venv .venv\n2. source .venv/bin/activate\n3. pip install -r requirements.txt\n4. pytest",
    },
    {
        "id": "5",
        "input": "Given the function, explain what it does:\n\ndef add(a, b):\n    return a + b",
        "output": "The function 'add' takes two arguments, a and b, and returns their sum.",
    },
]


# --- Formatting function ---
def formatting_func(
    example: Dict[str, str], prompt_style: str = "instruction"
) -> Tuple[str, str]:
    """
    Convert a dataset example into (prompt, completion) pair suitable for training an instruct-style LM.

    Args:
      example: dict with at least 'input' and 'output' keys.
      prompt_style: one of:
        - "instruction": explicit instruction + separator
        - "chat": a simple two-turn chat format (User / Assistant)
        - "single": a single concatenated string with a separator token

    Returns:
      (prompt, completion) where prompt is fed to the model and completion is the target continuation.
    """
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()

    if prompt_style == "chat":
        prompt = f"User: {inp}\nAssistant:"
        completion = f" {out}"
    elif prompt_style == "single":
        prompt = f"{inp}\n\n### Response:\n"
        completion = f"{out}"
    else:  # default to "instruction"
        prompt = f"Instruction:\n{inp}\n\nResponse:"
        completion = f" {out}"

    return prompt, completion


model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/SmolLM2-1.7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=60,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer_stats = trainer.train()
