import torch
import torch.nn as nn
import random

class RandomLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)

    def generate(self, start_token, max_length):
        current_token = start_token
        generated = [current_token]

        for _ in range(max_length - 1):
            current_token = torch.tensor([[current_token]])
            output = self(current_token)
            next_token = random.randint(0, output.size(-1) - 1)
            generated.append(next_token)
            current_token = next_token

        return generated

class SLMChatbot:
    def __init__(self, model, max_length):
        self.model = model
        self.max_length = max_length
        self.letters = ['s', 'l', 'm']

    def generate_response(self):
        sequence = self.model.generate(start_token=0, max_length=self.max_length)
        response = ''.join(self.letters[token % 3] for token in sequence[:3])
        return ''.join(random.choice([str.lower, str.upper])(c) for c in response)

vocab_size = 3  
embedding_dim = 16
hidden_dim = 32
max_length = 3  

model = RandomLanguageModel(vocab_size, embedding_dim, hidden_dim)
chatbot = SLMChatbot(model, max_length)

print("Small Language Model: Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Small Language Model: slm!")
        break
    response = chatbot.generate_response()
    print(f"Small Language Model: {response}")