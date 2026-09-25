import torch
import torch.nn as nn

class SoCLSTM(nn.Module):
    """Two-layer LSTM for SoC-drop prediction (Section 3.1 of the paper)."""
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()  # enforces non-negative SoC drop (Remark 2)

    def forward(self, x):
        # x: (batch, seq_len=50, features=4)
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last hidden state
        return self.relu(self.fc(out))
