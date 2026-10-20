import torch
from unsloth import FastLanguageModel
from ai_learning.core.config import config
from ai_learning.core.logging import setup_logging

logger = setup_logging()

# --- Configuration ---
PROMPT = "What is the largest planet in our solar system?"
MODEL_NAME = "unsloth/SmolLM2-1.7B-Instruct"
MAX_SEQ_LENGTH = 8000
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 8000
TEMPERATURE = 1
TOP_P = 0.99
DO_SAMPLE = True
INSPECT_MODEL_AND_TOKENIZER = True

def _inspect_tokenizer(tokenizer):
    """Print tokenizer structure: type, vocab size, special tokens, and a small encode/decode example."""
    print("\n" + "=" * 60)
    print("TOKENIZER INSPECTION")
    print("=" * 60)
    print(f"  Type: {type(tokenizer).__module__}.{type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    for name in ("pad_token", "eos_token", "bos_token", "unk_token"):
        attr = name.replace("_token", "_token_id")
        val = getattr(tokenizer, attr, None)
        token_str = getattr(tokenizer, name, None)
        print(f"  {name}: id={val!r}  repr={token_str!r}")
    vocab = getattr(tokenizer, "get_vocab", None) or (lambda: {})
    if callable(vocab):
        vocab = vocab()
    if vocab:
        by_id = sorted((id_, token) for token, id_ in vocab.items())
        for tid, tstr in by_id[:8]:
            print(f"    {tid}: {repr(tstr)}")
    example = "Hello world"
    enc = tokenizer.encode(example, add_special_tokens=False)
    dec = tokenizer.decode(enc)
    print(f"  Encode('{example}'): {enc}")
    print(f"  Decode(enc): {repr(dec)}")
    print("=" * 60 + "\n")

def _inspect_model(model, tokenizer):
    """Print model structure: type, config, param count, and top-level layer names."""
    print("\n" + "=" * 60)
    print("MODEL INSPECTION")
    print("=" * 60)
    print(f"  Type: {type(model).__module__}.{type(model).__name__}")
    conf = getattr(model, "config", None)
    if conf is not None:
        print("  Config (main keys):")
        for key in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "max_position_embeddings"):
            if hasattr(conf, key):
                print(f"    {key}: {getattr(conf, key)}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total:,}  (trainable: {trainable:,})")
    print(f"  Approx. size @ 4 bytes: {total * 4 / 1e9:.2f} GB  |  @ 2 bytes (bf16): {total * 2 / 1e9:.2f} GB")
    print("  Top-level module names:")
    for name, _ in list(model.named_children())[:20]:
        print(f"    - {name}")

    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        print("  Inner .model top-level (first 12):")
        for name, _ in list(inner.named_children())[:12]:
            print(f"    - {name}")

    embed = getattr(model, "embed_tokens", None) or (getattr(inner, "embed_tokens", None) if inner is not None else None)
    if embed is not None:
        print("  embed_tokens:")
        print(f"    type: {type(embed).__module__}.{type(embed).__name__}")
        if hasattr(embed, "weight"):
            w = embed.weight
            print(f"    weight.shape: {tuple(w.shape)}  (vocab_size, hidden_size)")
            print(f"    weight.dtype: {w.dtype}")
            ids = tokenizer.encode("Hello world", add_special_tokens=False)
            if ids:
                id_tensor = torch.tensor([ids], device=w.device)
                with torch.no_grad():
                    vecs = embed(id_tensor)
                print('  Embedding vectors for "Hello world":')
                print(f"    token IDs: {ids}")
                print(f"    output shape: {tuple(vecs.shape)}  (batch=1, num_tokens, hidden_size)")
                for i, tid in enumerate(ids):
                    v = vecs[0, i]
                    head = v[:8].tolist()
                    rest = v.numel() - 8
                    print(f"    token {i} (id={tid}): first 8 dims {[round(x, 5) for x in head]}{f' ... +{rest} more' if rest > 0 else ''}")
    print("=" * 60 + "\n")

def run_inference():
    # GPU check
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs a CUDA GPU. PyTorch did not detect one.")

    logger.info("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    if INSPECT_MODEL_AND_TOKENIZER:
        _inspect_tokenizer(tokenizer)
        _inspect_model(model, tokenizer)

    FastLanguageModel.for_inference(model)

    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    logger.info(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    if INSPECT_MODEL_AND_TOKENIZER:
        with torch.no_grad():
            model_output = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            logits = model_output.logits

        print("\n" + "=" * 60)
        print("LM_HEAD LOGITS INSPECTION")
        print("=" * 60)
        print(f"  logits shape: {tuple(logits.shape)}  (batch, seq_len, vocab_size)")
        last_logits = logits[0, -1, :]
        print(f"  Last position logits shape: {tuple(last_logits.shape)}  (one score per vocab word)")
        print(f"  Min score: {last_logits.min().item():.4f}   Max score: {last_logits.max().item():.4f}")
        probs = torch.nn.functional.softmax(last_logits.float(), dim=0)
        top_k = 20
        top_scores, top_ids = torch.topk(last_logits, top_k)
        print(f"\n  Top {top_k} predicted next tokens (from {last_logits.shape[0]} candidates):")
        print(f"  {'Rank':<6} {'ID':<8} {'Token':<20} {'Logit':>10} {'Probability':>12}")
        print(f"  {'-' * 6} {'-' * 8} {'-' * 20} {'-' * 10} {'-' * 12}")
        for rank, (score, tid) in enumerate(zip(top_scores, top_ids), 1):
            token_str = tokenizer.decode([tid.item()])
            prob = probs[tid.item()].item()
            print(f"  {rank:<6} {tid.item():<8} {repr(token_str):<20} {score.item():>10.4f} {prob:>11.4%}")

        greedy_id = last_logits.argmax().item()
        greedy_token = tokenizer.decode([greedy_id])
        print(f"\n  Greedy pick (argmax): ID={greedy_id}  token={repr(greedy_token)}  score={last_logits[greedy_id].item():.4f}")
        print("=" * 60 + "\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    response_ids = outputs[0][prompt_length:]
    answer = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    print("\n--- Prompt ---")
    print(PROMPT)
    print("\n--- Model answer ---")
    print(answer)
    print()

if __name__ == "__main__":
    run_inference()
