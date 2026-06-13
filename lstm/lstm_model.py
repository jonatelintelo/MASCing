import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class MoETraceClassifierLinear(nn.Module):
    def __init__(self, num_total_experts, num_layers, embed_dim=16, hidden_dim=64):
        super().__init__()
        
        self.num_total_experts = num_total_experts
        
        # 1. NEW: Normalize raw logits to prevent LSTM gate saturation
        self.layer_norm = nn.LayerNorm(num_total_experts)
        
        # 2. Linear projection
        self.expert_projection = nn.Linear(num_total_experts, embed_dim)
        
        # Dynamic Input Size Calculation
        self.lstm_input_size = num_layers * embed_dim
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_dim, batch_first=True)
        
        # 3. Output 1 logit for BCE loss
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x_masked, lengths):
        batch_size, max_seq_len, _, _ = x_masked.shape

        # Normalize the raw logits across the expert dimension
        x_norm = self.layer_norm(x_masked)

        # Project the normalized logits into the embedding space
        x_emb = self.expert_projection(x_norm)

        # Flatten layers and embedding dimensions for the LSTM
        x_flat = x_emb.view(batch_size, max_seq_len, -1)

        packed_input = pack_padded_sequence(x_flat, lengths, batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.lstm(packed_input)

        return self.classifier(ht[-1])