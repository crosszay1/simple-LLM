import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sequence_len = 5

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TinyLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.rnn(out)
        out = self.fc(out)
        return out


def load_tokenizer():
    try:
        import tiktoken
    except Exception as e:
        print("tiktoken is required but not installed:", e)
        sys.exit(1)
    return tiktoken.get_encoding("o200k_base")


def generate_text_from_model(model, tokenizer, prompt, max_length=128, device=None):
    if device is None:
        device = get_device()
    model.eval()
    prompt_tokens = tokenizer.encode(prompt)[-sequence_len:]
    generated = prompt_tokens.copy()
    with torch.no_grad():
        for _ in range(max_length):
            current = [generated[-sequence_len:]]
            current = torch.tensor(current, dtype=torch.long, device=device)
            output = model(current)
            preds = output[:, -1, :]
            probs = F.softmax(preds, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
    return tokenizer.decode(generated)


if __name__ == "__main__":
    device = get_device()
    tokenizer = load_tokenizer()

    # Try common checkpoint filenames
    candidates = ["tiny_llm_checkpoint.pt", "tiny_llm_weights.pt", "model.pt", "model.pt"]
    ckpt = None
    for c in candidates:
        if os.path.exists(c):
            ckpt = c
            break

    if ckpt is None:
        print("No checkpoint or weights file found. Train first and save weights (see train.py).")
        sys.exit(1)

    # We need vocab size and architecture; these must match training
    vocab_size = tokenizer.n_vocab
    embed_size = 128
    hidden_size = 256

    model = TinyLLM(vocab_size, embed_size, hidden_size).to(device)

    print(f"Loading weights from {ckpt}...")
    data = torch.load(ckpt, map_location=device)
    if isinstance(data, dict) and "model_state_dict" in data:
        model.load_state_dict(data["model_state_dict"])
    else:
        # assume it's a raw state_dict
        model.load_state_dict(data)

    print('Model ready. Type prompts (exit to quit).')
    while True:
        try:
            prompt = input("Prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            break
        out = generate_text_from_model(model, tokenizer, prompt, max_length=128, device=device)
        print(out)
