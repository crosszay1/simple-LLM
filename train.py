import tiktoken #cool tokenizer by openai
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim


#        config stuffs        #
sequence_len = 5



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



#For now we just train on one data piece
data = ""
with open("data/CIA_World_Fact_Sheet_2006.txt", "r") as f:
  data = f.read()

encoding = tiktoken.get_encoding("o200k_base")


X, Y = generate_training_data(data, sequence_len, encoding)


tensor_X = torch.tensor(X, dtype = torch.long)
tensor_y = torch.tensor(Y, dtype = torch.long)

tensor_X = tensor_X.to("cuda")
tensor_y = tensor_y.to("cuda")

tensor_X.shape, tensor_y.shape



dataset = TensorDataset(tensor_X, tensor_y)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


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
).to("cuda")


num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params}")



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)






















print("Reached training loop")
n_epochs = 40

for epoch in range(n_epochs):
  epoch_loss = 0
  for X, y in dataloader:
    optimizer.zero_grad()
    outputs = model(X)
    outputs = outputs.view(-1, encoding.n_vocab)
    y = y.view(-1)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()

  avg_loss = epoch_loss / len(dataloader)
  print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')
