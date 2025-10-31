# chatbot_better.py
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./fine_tuned_model"   # keep as is if you fine-tuned locally

SYSTEM_PRIME = (
    "System: You are a calm, empathetic mental-wellbeing assistant. "
    "Validate feelings, ask one clarifying question if needed, suggest one small next step. "
    "Do not diagnose. If crisis is indicated, advise contacting local emergency services."
)

def load_model(model_dir: str = MODEL_DIR, device: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(use_device).eval()
    model.generation_config.pad_token_id = tok.eos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    return tok, model, use_device

def left_trim(ids: torch.Tensor, mask: torch.Tensor, keep: int):
    if ids.shape[-1] <= keep:
        return ids, mask
    return ids[:, -keep:], mask[:, -keep:]

def format_dialogue(turns):
    """
    Use explicit role tags the model can latch onto.
    Example:
      System: ...
      User: ...
      Assistant: ...
    """
    return "\n".join(turns) + "\nAssistant:"

def main():
    tok, model, device = load_model()
    ctx_window = getattr(model.config, "n_positions", 1024)
    max_new = 128
    budget = max(32, ctx_window - max_new - 16)

    print("Chat ready. Type 'exit' to quit, 'clear' to reset memory.\n")

    # rolling token history
    hist_ids, hist_mask = None, None

    # seed/system message
    turns = [SYSTEM_PRIME]

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        low = user.lower()
        if low in {"exit", "quit", "q"}:
            print("Bot: Take care. 👋")
            break
        if low in {"clear", "/clear"}:
            turns = [SYSTEM_PRIME]
            hist_ids, hist_mask = None, None
            print("Bot: Memory cleared.\n")
            continue

        # add the user line with an explicit tag
        turns.append(f"User: {user}")

        # build the prompt string with tags
        prompt = format_dialogue(turns)

        enc = tok(prompt, return_tensors="pt", truncation=True, padding=False)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        # trim to context budget
        input_ids, attn_mask = left_trim(input_ids, attn_mask, budget)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.7,         # a touch lower for stability
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.12, # reduces loops/clichés
                no_repeat_ngram_size=3,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )

        new_tokens = out[:, input_ids.shape[-1]:]
        reply = tok.decode(new_tokens[0], skip_special_tokens=True).strip()

        # cut at a sensible stop if the model emits extra roles
        for stop in ["\nUser:", "\nSystem:", "\nAssistant:"]:
            if stop in reply:
                reply = reply.split(stop)[0].strip()

        if not reply:
            reply = "It sounds like you’re going through a tough moment. Could you share a bit more about when the anxiety shows up most?"

        print(f"Bot: {reply}\n")

        # store the assistant turn explicitly
        turns.append(f"Assistant: {reply}")

if __name__ == "__main__":
    main()
