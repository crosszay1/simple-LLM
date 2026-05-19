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