import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, latent_dim=4, num_layers=1, dropout_prob=0.1):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        # Added LayerNorm for training stability
        self.encoder_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # Dropout is only applied between layers, so it's ignored if layers=1
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            hidden_dim,
            input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

    def forward(self, x):
        # x shape: [batch, seq, features]
        x = self.encoder_norm(x)

        # Encoder Pass
        _, (h, _) = self.encoder(x)

        # Compressing to Latent Space (The "Thought")
        # h[-1] contains the final hidden state of the last layer
        latent = self.hidden2latent(h[-1])  # Shape: [batch, latent_dim]

        # Decoder Pass
        # 1. Expand latent back to hidden size
        h_decoded = self.latent2hidden(latent)

        # 2. Repeat hidden state for every time step in the sequence
        # Shape becomes: [batch, seq_len, hidden_dim]
        h_decoded_repeated = h_decoded.unsqueeze(1).repeat(1, x.size(1), 1)

        # 3. Reconstruct
        reconstruction, _ = self.decoder(h_decoded_repeated)

        return reconstruction, latent


class SupplyChainBrain:
    def __init__(self, input_features=1, seq_length=30):
        self.seq_length = seq_length
        self.input_features = input_features

        # Device management (runs on GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize with num_layers=1 to match original simple architecture
        self.model = LSTMAutoencoder(input_dim=input_features, num_layers=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()

        self.is_trained = False

    def _to_tensor(self, data_buffer: List[float]) -> torch.Tensor:
        """Helper to convert list buffer to appropriate tensor."""
        # Ensure input is [Batch=1, Seq, Features]
        # First unsqueeze adds Batch dim, second adds Feature dim
        return torch.tensor(data_buffer, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(self.device)

    def learn(self, data_buffer: List[float]) -> float:
        """
        Standard calibration training (Gentle adaptation).
        Used during the initialization phase or periodic background updates.
        """
        if len(data_buffer) != self.seq_length:
            return 0.0

        self.model.train()
        tensor_data = self._to_tensor(data_buffer)

        # 10 Epochs for normal background learning
        final_loss = 0.0
        for _ in range(10):
            self.optimizer.zero_grad()
            recon, _ = self.model(tensor_data)
            loss = self.criterion(recon, tensor_data)
            loss.backward()
            self.optimizer.step()
            final_loss = loss.item()

        self.is_trained = True
        return final_loss

    def force_learn(self, sequence: List[float]) -> float:
        """
        EXCEPTION HANDLING: Aggressively learns a specific sequence.
        Triggered when the Human Operator marks an anomaly as 'Normal'.
        """
        if len(sequence) != self.seq_length:
            return 0.0

        self.model.train()
        tensor_data = self._to_tensor(sequence)

        # 50 Epochs to FORCE the model to accept this pattern immediately
        final_loss = 0.0
        for _ in range(50):
            self.optimizer.zero_grad()
            recon, _ = self.model(tensor_data)
            loss = self.criterion(recon, tensor_data)
            loss.backward()
            self.optimizer.step()
            final_loss = loss.item()

        return final_loss

    def detect(self, sequence: List[float]) -> Tuple[float, np.ndarray]:
        """
        Returns Anomaly Score (MSE) and the Latent 'Thought' vector.
        """
        if not self.is_trained or len(sequence) != self.seq_length:
            return 0.0, np.zeros(4)

        self.model.eval()
        with torch.no_grad():
            tensor_input = self._to_tensor(sequence)
            recon, latent = self.model(tensor_input)

            # MSE Loss is the Anomaly Score
            loss = self.criterion(recon, tensor_input)

        return loss.item(), latent.cpu().numpy()[0]