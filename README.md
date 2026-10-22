# 🧠 AI Learning Project

A professional multi-module Python repository for exploring Large Language Models (LLMs), Retrieval Augmented Generation (RAG), Agentic Workflows, and Model Fine-tuning.

## 🚀 Project Structure

This project uses a `src` layout for professional package management.

```text
ai-learning/
├── src/ai_learning/         # Main package
│   ├── core/                # Shared infrastructure (Logging, Config, Clients)
│   └── modules/             # Learning modules
│       ├── ollama_cli/      # Ollama interactive tools
│       ├── rag/             # RAG pipeline (PDF ingestion & Querying)
│       ├── crewai/          # Multi-agent orchestration
│       ├── huggingface/     # Image generation and HF models
│       └── unsloth/         # Efficient LLM fine-tuning (AMD/ROCm optimized)
├── data/                    # Shared datasets and RAG documents
├── notebooks/               # Experimental Jupyter notebooks
└── tests/                   # Module test suite
```

## 🛠️ Installation & Hardware Setup

This project is optimized for **AMD GPUs (ROCm)**. We use `uv` for lightning-fast dependency management.

### 1. Install `uv`
If you haven't installed `uv` yet:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Environment
To ensure the ROCm GPU overrides and `uv` cache settings are applied automatically, add the `.env` loader to your virtual environment:
```bash
uv sync
echo 'export $(cat .env | xargs)' >> .venv/bin/activate
source .venv/bin/activate
```

### 3. Sync Dependencies
The project uses dependency groups to keep the environment lean.

**Base Installation (Fast):**
```bash
uv sync
```

**Full Installation (Includes Heavy ML/ROCm stack):**
```bash
uv sync --all-groups
```

**Crucial: Install project in editable mode**
To allow the modules to find each other, you must install the project as a package:
```bash
uv pip install -e .
```

## 💻 Usage

All modules are accessible via a unified CLI.

### Ollama Chat
Start an interactive chat session with your local LLM:
```bash
uv run python -m ai_learning.cli chat
```

### RAG Bot
Query your indexed PDF documents:
```bash
uv run python -m ai_learning.cli rag
```

### Model Training (Unsloth)
Fine-tune a model using the optimized Unsloth pipeline:
```bash
uv run python -m ai_learning.cli train
```

### Model Evaluation (Unsloth)
Evaluate a inference model using the Unsloth evaluation pipeline:
```bash
uv run python -m ai_learning.cli unsloth
```

### Image Generation (HuggingFace)
Generate images using Stable Diffusion:
```bash
uv run python -m ai_learning.cli generate
```

### CrewAI Multi-Agent Orchestration
Recipe and Nutrition agents for multi-agent orchestration:
```bash
uv run streamlit run src/ai_learning/modules/crewai/agents.py
```

## ⚙️ Configuration
Environment variables are managed via a `.env` file in the root directory.
- `OLLAMA_HOST`: Default is `http://localhost:11434`
- `DEFAULT_MODEL`: The model used across the project (e.g., `qwen3.5:0.8b`)
- `HSA_OVERRIDE_GFX_VERSION`: Required for AMD GPU compatibility.

## 📚 Learning Path
- **Module 01**: Basics of LLM interaction via Ollama.
- **Module 02**: Building a RAG pipeline with ChromaDB and LangChain.
- **Module 03**: Multi-agent orchestration with CrewAI.
- **Module 04**: Generative AI with Hugging Face Diffusers.
- **Module 05**: Efficient fine-tuning with Unsloth and ROCm.
