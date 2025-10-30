import os
import argparse
import threading
from typing import Optional, List
import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
except ImportError:  # older transformers: fall back to non-streaming
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TextIteratorStreamer = None

MODEL_DIR = "./fine_tuned_model"


def load_model(model_dir: str = MODEL_DIR, device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    # GPT-2/DialoGPT usually lack a pad token; reuse EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pick device and dtype
    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = device or auto_device
    torch_dtype = torch.float16 if (device == "cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    # Avoid pad/eos warnings during generation
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    return tokenizer, model, device


def compute_context_limit(model, max_new_tokens: int, safety_margin: int = 16) -> int:
    # DialoGPT/GPT2 commonly expose n_positions as context window
    n_pos = getattr(model.config, "n_positions", 1024)
    keep = max(32, n_pos - max_new_tokens - safety_margin)
    return keep


def build_prompt_tokens(tokenizer, turns: list[str], max_context_tokens: int) -> torch.Tensor:
    # Join turns with eos as separator and let tokenizer handle truncation to context
    text = (tokenizer.eos_token or "\n").join(turns) + (tokenizer.eos_token or "")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_context_tokens)
    return enc["input_ids"]


def stream_generate(
    tokenizer,
    model,
    device: str,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
):
    """Generate with streaming; yields decoded text chunks and returns full reply when done."""
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    gen_kwargs = dict(
        input_ids=input_ids.to(device),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    def _worker():
        with torch.no_grad():
            model.generate(**gen_kwargs)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    chunks = []
    for token_text in streamer:
        chunks.append(token_text)
        yield token_text  # stream out


def main():
    parser = argparse.ArgumentParser(description="Interactive chatbot with short-term memory")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Path or HF id of the model to load")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="auto/cuda/cpu")
    args = parser.parse_args([]) if os.environ.get("INLINE_ARGS", "0") == "1" else parser.parse_args()

    print("Loading model…")
    chosen_device = None if args.device == "auto" else args.device
    tokenizer, model, device = load_model(args.model_dir, chosen_device)

    max_context_tokens = compute_context_limit(model, args.max_new_tokens)

    print("Chat ready. Commands: 'exit' to quit, 'clear' to reset memory.\n")

    # Conversation memory as plain turns (joined with EOS for tokenization)
    turns: List[str] = []

    # One-time priming to steer tone
    priming = (
        "You are a calm, empathetic assistant for mental wellbeing. "
        "Be supportive, non-judgmental, and concise."
    )
    turns.append(priming)

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            lo = user.lower()
            if lo in {"exit", "quit", "q"}:
                print("Bot: Take care. 👋")
                break
            if lo in {"clear", "/clear"}:
                turns = [priming]
                print("Bot: Memory cleared.")
                continue

            # Add user turn and build prompt tokens with truncation to fit context window
            turns.append(user)
            input_ids = build_prompt_tokens(tokenizer, turns, max_context_tokens)
            input_len = input_ids.shape[-1]

            # Stream the model's reply
            print("Bot:", end=" ", flush=True)
            reply_chunks = []
            for chunk in stream_generate(
                tokenizer=tokenizer,
                model=model,
                device=device,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            ):
                reply_chunks.append(chunk)
                print(chunk, end="", flush=True)

            reply = ("".join(reply_chunks)).strip()
            print("\n")

            # Remember the assistant reply for short-term memory
            turns.append(reply)

    except KeyboardInterrupt:
        print("\nBot: Session ended. 👋")


if __name__ == "__main__":
    main()

