import torch
import torch.nn as nn

class InferenceModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(InferenceModel, self).__init__()
        self._hidden_layers = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh()
        )
        self._logits = nn.Linear(100, output_size)
        self._value_branch_separate = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh()
        )
        self._value_branch = nn.Linear(100, 1)

    def forward(self, x):
        hidden_output = self._hidden_layers(x)
        logits = self._logits(hidden_output)
        value_hidden_output = self._value_branch_separate(x)
        value = self._value_branch(value_hidden_output)
        
        return logits, value