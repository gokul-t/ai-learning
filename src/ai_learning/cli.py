import typer
from ai_learning.modules.ollama_cli import llm
from ai_learning.modules.rag import run_rag_cli
from ai_learning.modules.huggingface import generation
from ai_learning.modules.unsloth import trainer, inference

app = typer.Typer(help="AI Learning Project CLI")

@app.command()
def chat():
    """Start an interactive chat with the default Ollama model."""
    llm.run_chat_cli()

@app.command()
def rag():
    """Start the PDF RAG bot."""
    run_rag_cli()

@app.command()
def train():
    """Start training a model using Unsloth."""
    trainer.train()

@app.command()
def unsloth():
    """Run inference using the Unsloth model."""
    inference.run_inference()

@app.command()
def generate():
    """Run inference using the Unsloth model."""
    generation.main()

if __name__ == "__main__":
    app()
