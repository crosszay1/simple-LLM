import tiktoken #cool tokenizer by openai
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path


#        config stuffs        #
sequence_len = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_training_samples = 100_000



def user_input():
    question = input("Enter your prompt: ")
    return question

def construct_prompt(question):
    prompt = f"""
    Question: {question}\n
    Answer:
    """
    return prompt


def generate_training_data(data, n, tokenizer):
    tokens = tokenizer.encode(data)
    X = []
    y = []
    for i in range(len(tokens) - n):
      X.append(tokens[i : n + i])
      y.append(tokens[i + 1 : n + i + 1])

    return [X, y]
#X = input, y = output
#Model tries to predict y given X.



def load_training_data(data_dir):
  parts = []
  for file_path in sorted(Path(data_dir).rglob("*")):
    if file_path.is_file():
      parts.append(file_path.read_text(encoding="utf-8"))
  return "\n\n".join(parts)

#Multiple files!!!
data = load_training_data("data")

encoding = tiktoken.get_encoding("o200k_base")


X, Y = generate_training_data(data, sequence_len, encoding)

if max_training_samples is not None:
  X = X[:max_training_samples]
  Y = Y[:max_training_samples]


tensor_X = torch.tensor(X, dtype = torch.long)
tensor_y = torch.tensor(Y, dtype = torch.long)
tensor_X.shape, tensor_y.shape



dataset = TensorDataset(tensor_X, tensor_y)
dataloader = DataLoader(
  dataset,
  batch_size=256,
  shuffle=True,
  pin_memory=(device.type == "cuda"),
)


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



embed_size = 128
hidden_size = 256

model = TinyLLM(encoding.n_vocab, 
                embed_size, hidden_size
).to(device)


num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params}")



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)








def generate_text(prompt, tokenizer, max_length = 50):
  prompt = tokenizer.encode(prompt)[-sequence_len:]
  generated = prompt.copy()
  with torch.no_grad():
    for _ in range(max_length):
      current = [generated[-sequence_len:]]
      current = torch.tensor(current, dtype=torch.long, device=device)
      output = model(current)
      predictions = output[:, -1, :]
      probabilities = F.softmax(predictions, dim=-1)
      next_token = torch.multinomial(probabilities, num_samples=1).item()
      generated.append(next_token)
  print(encoding.decode(generated))



print("Reached training loop")
n_epochs = 40
print(f"Training on {len(dataset):,} samples across {len(dataloader):,} batches on {device}.")

for epoch in range(n_epochs):
  epoch_loss = 0
  for batch_index, (X, y) in enumerate(dataloader, start=1):
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    optimizer.zero_grad()
    outputs = model(X)
    outputs = outputs.view(-1, encoding.n_vocab)
    y = y.view(-1)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()

    if batch_index % 100 == 0:
      print(f"Epoch {epoch + 1}/{n_epochs}, batch {batch_index}/{len(dataloader)}, loss {loss.item():.4f}")

  avg_loss = epoch_loss / len(dataloader)
  print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')


print("Training complete. Testing model with prompt...")
generate_text("Hello", encoding, max_length=256)
print("Test complete.")
print("Saving model...")
torch.save(model.state_dict(), "model.pt")
print("Done. Exiting...")
exit(1)